#include <cuda.h>
#include <cuda_runtime.h>
#ifndef __wrap_t__
#define __wrap_t__
template <typename scalar_t> struct wrap_t{
    int b, l, d, n;
    int stepb, stepl;
    scalar_t* p;
};
#endif//__wrap_t__

#ifndef IDX5D
#define IDX5D(shape) ((blockIdx.x % shape.b) * shape.stepb + (blockIdx.y % shape.d) * shape.n + threadIdx.x % shape.n)
#define Ptr5D(shape) (shape.p + IDX5D(shape))
#endif//IDX5D

#define atomAdd atomicAdd
#ifndef GROUP_SIZE
#define GROUP_SIZE 1023
#endif//

namespace { namespace device {
    template <typename scalar_t> __global__ void causalScan4d_Forward(
        const wrap_t<scalar_t> shapeX,
        const wrap_t<scalar_t> shapeZ,
        const wrap_t<scalar_t> shapeA,
        const wrap_t<scalar_t> shapeB,
        const wrap_t<scalar_t> shapeC,
        const wrap_t<scalar_t> shapeO
    )
    {
        scalar_t * pX = Ptr5D(shapeX);
        scalar_t * pZ = Ptr5D(shapeZ);
        scalar_t * pA = Ptr5D(shapeA);
        scalar_t * pB = Ptr5D(shapeB);
        scalar_t * pC = Ptr5D(shapeC);
        scalar_t * pO = Ptr5D(shapeO);
        scalar_t * pH = pZ;
        scalar_t zh = *pZ;
        int i = 0;
        while(i++<shapeO.l) {
            if( i % GROUP_SIZE == 0 ) {
                pH += shapeZ.stepl;
                *pH = zh;
            }
            zh = (*pA) * zh + (*pB) * (*pX);
            atomAdd(pO, ((*pC) * zh));
            pX += shapeX.stepl;
            pA += shapeA.stepl;
            pB += shapeB.stepl;
            pC += shapeC.stepl;
            pO += shapeO.stepl;
        }
        pZ[(shapeZ.l-1)*shapeZ.stepl] = zh;
    }

    template <typename scalar_t> __global__ void causalScan4d_Backward(
        scalar_t * pX,
        scalar_t * pZ,
        scalar_t * pA,
        scalar_t * pB,
        scalar_t * pC,
        const wrap_t<scalar_t> gradO,
        const wrap_t<scalar_t> gradX,
        const wrap_t<scalar_t> gradZ,
        const wrap_t<scalar_t> gradA,
        const wrap_t<scalar_t> gradB,
        const wrap_t<scalar_t> gradC
    )
    {
        int length = gradO.l;
        int sx = IDX5D(gradX);
        int sz = IDX5D(gradZ);
        int sa = IDX5D(gradA);
        int sb = IDX5D(gradB);
        int sc = IDX5D(gradC);
        pX += sx;
        pZ += sz;
        pA += sa;
        pB += sb;
        pC += sc;
        scalar_t * pGradO = gradO.p + sx;
        scalar_t * pGradX = gradX.p + sx;
        scalar_t * pGradZ = gradZ.p + sz;
        scalar_t * pGradA = gradA.p + sa;
        scalar_t * pGradB = gradB.p + sb;
        scalar_t * pGradC = gradC.p + sc;

        scalar_t gradh = 0.0;
        scalar_t zhs[GROUP_SIZE+1];
        int groups = (length + GROUP_SIZE - 1) / GROUP_SIZE;
        for(int igroups=groups-1; igroups>=0; igroups--){
            int ibegin = igroups * GROUP_SIZE;
            int group_length = (igroups==groups-1)?(length-ibegin):GROUP_SIZE;

            scalar_t * pIX = pX + ibegin*gradX.stepl;
            scalar_t * pIA = pA + ibegin*gradA.stepl;
            scalar_t * pIB = pB + ibegin*gradB.stepl;
            zhs[0] = pZ[igroups*gradZ.stepl];
            for(int i=0; i<group_length; i++) {
                zhs[i+1] = (*pIA) * zhs[i] + (*pIB) * (*pIX);
                pIA += gradA.stepl;
                pIB += gradB.stepl;
                pIX += gradX.stepl;
            }

            int iend = ibegin + group_length;
            scalar_t * pIC = pC + iend * gradC.stepl;
            scalar_t * pIGradO = pGradO + iend * gradO.stepl;
            scalar_t * pIGradX = pGradX + iend * gradX.stepl;
            scalar_t * pIGradA = pGradA + iend * gradA.stepl;
            scalar_t * pIGradB = pGradB + iend * gradB.stepl;
            scalar_t * pIGradC = pGradC + iend * gradC.stepl;
            while(group_length-->0) {
                pIA -= gradA.stepl;
                pIB -= gradB.stepl;
                pIX -= gradX.stepl;
                pIC -= gradC.stepl;
                pIGradO -= gradO.stepl;
                pIGradA -= gradA.stepl;
                pIGradB -= gradB.stepl;
                pIGradX -= gradX.stepl;
                pIGradC -= gradC.stepl;

                atomAdd(pIGradC, (*pIGradO) * zhs[group_length+1]);
                gradh += (*pIGradO) * (*pC);
                atomAdd(pIGradB, gradh * (*pX));
                atomAdd(pIGradX, gradh * (*pB));
                atomAdd(pIGradA, zhs[group_length] * gradh);
                gradh *= (*pIA);
            }
        }
        *pGradZ = gradh;
    }
}}

#undef atomAdd
#define __PYBINDED__
#include "./CausalScan4d.cpp"
torch::Tensor causalScan4d_cuda_Forward(
    torch::Tensor X, 
    torch::Tensor Z, 
    torch::Tensor A,
    torch::Tensor B, 
    torch::Tensor C
) {
    if(!X.is_cuda()) {
        return causalScan4d_cpu_Forward(X, Z, A, B, C);
    }

    auto O = torch::zeros_like(X);
    AT_DISPATCH_FLOATING_TYPES(O.type(), "causalScan4d_Forward", ([&] {
        wrap_t<scalar_t> shapeX = SHAPE5D(X);
        wrap_t<scalar_t> shapeZ = SHAPE5D(Z);
        wrap_t<scalar_t> shapeA = SHAPE5D(A);
        wrap_t<scalar_t> shapeB = SHAPE5D(B);
        wrap_t<scalar_t> shapeC = SHAPE5D(C);
        wrap_t<scalar_t> shapeO = SHAPE5D(O);
        int threads = shapeZ.n;
        const dim3 blocks(shapeZ.b, shapeZ.d);    
        device::causalScan4d_Forward<scalar_t><<<blocks, threads>>>(
            shapeX,
            shapeZ,
            shapeA,
            shapeB,
            shapeC,
            shapeO
        );
    }));
    return O;
}

std::vector<torch::Tensor> causalScan4d_cuda_Backward(
    torch::Tensor gradO,
    torch::Tensor X,
    torch::Tensor Z,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C
) {
    if(!gradO.is_cuda()) {
        return causalScan4d_cpu_Backward(gradO, X, Z, A, B, C);
    }
    auto gradX = torch::zeros_like(X);
    auto gradZ = torch::zeros_like(Z);
    auto gradA = torch::zeros_like(A);
    auto gradB = torch::zeros_like(B);
    auto gradC = torch::zeros_like(C);
    AT_DISPATCH_FLOATING_TYPES(gradO.type(), "causalScan4d_Backward", ([&] {
        wrap_t<scalar_t> deltaO = SHAPE5D(gradO);
        wrap_t<scalar_t> deltaX = SHAPE5D(gradX);
        wrap_t<scalar_t> deltaZ = SHAPE5D(gradZ);
        wrap_t<scalar_t> deltaA = SHAPE5D(gradA);
        wrap_t<scalar_t> deltaB = SHAPE5D(gradB);
        wrap_t<scalar_t> deltaC = SHAPE5D(gradC);
        int threads = deltaZ.n;
        const dim3 blocks(deltaZ.b, deltaZ.d);
        device::causalScan4d_Backward<scalar_t><<<blocks, threads>>>(
            (scalar_t*)X.data_ptr(),
            (scalar_t*)Z.data_ptr(),
            (scalar_t*)A.data_ptr(),
            (scalar_t*)B.data_ptr(),
            (scalar_t*)C.data_ptr(),
            deltaO,
            deltaX,
            deltaZ,
            deltaA,
            deltaB,
            deltaC
        );
    }));
    return {gradX, gradZ, gradA, gradB, gradC};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &causalScan4d_cuda_Forward, "");
    m.def("backward", &causalScan4d_cuda_Backward, "");
}