#include <torch/extension.h>
#include <vector>

#ifndef __wrap_t__
#define __wrap_t__
template <typename scalar_t> struct wrap_t{
    int b, l, d, n;
    int stepb, stepl;
    scalar_t* p;
};
#endif//__wrap_t__
typedef struct{
    int x, y;
}INDICS;

#ifndef IDX5D
#define IDX5D(shape) ((blockIdx.x % shape.b) * shape.stepb + (blockIdx.y % shape.d) * shape.n + threadIdx.x % shape.n)
#define Ptr5D(shape) (shape.p + IDX5D(shape))
#endif//IDX5D

#ifndef atomAdd
#define atomAdd(p,b) (*(p) = *(p) + (b))
#endif//atomAdd

#ifndef GROUP_SIZE
#define GROUP_SIZE 1023
#endif//

namespace { namespace device {
    template <typename scalar_t> void causalScan4d_cpu_Forward(
        const wrap_t<scalar_t> shapeX,
        const wrap_t<scalar_t> shapeZ,
        const wrap_t<scalar_t> shapeA,
        const wrap_t<scalar_t> shapeB,
        const wrap_t<scalar_t> shapeC,
        const wrap_t<scalar_t> shapeO,
        const INDICS& blockIdx,
        const INDICS& threadIdx
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
            zh = (*pA) * zh + (*pB) * (*pX);
            atomAdd(pO, ((*pC) * zh));
            if( i % GROUP_SIZE == 0 ) {
                pH += shapeZ.stepl;
                *pH = zh;
            }
            pX += shapeX.stepl;
            pA += shapeA.stepl;
            pB += shapeB.stepl;
            pC += shapeC.stepl;
            pO += shapeO.stepl;
        }
        pZ[(shapeZ.l-1)*shapeZ.stepl] = zh;
    }

    template <typename scalar_t> void causalScan4d_cpu_Backward(
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
        const wrap_t<scalar_t> gradC,
        const INDICS& blockIdx,
        const INDICS& threadIdx
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
                pIX += gradX.stepl;
                pIA += gradA.stepl;
                pIB += gradB.stepl;
            }

            int iend = ibegin + group_length;
            scalar_t * pIC = pC + iend * gradC.stepl;
            scalar_t * pIGradO = pGradO + iend * gradO.stepl;
            scalar_t * pIGradX = pGradX + iend * gradX.stepl;
            scalar_t * pIGradA = pGradA + iend * gradA.stepl;
            scalar_t * pIGradB = pGradB + iend * gradB.stepl;
            scalar_t * pIGradC = pGradC + iend * gradC.stepl;
            while(group_length-->0) {
                pIX -= gradX.stepl;
                pIA -= gradA.stepl;
                pIB -= gradB.stepl;
                pIC -= gradC.stepl;
                pIGradO -= gradO.stepl;
                pIGradX -= gradX.stepl;
                pIGradA -= gradA.stepl;
                pIGradB -= gradB.stepl;
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


#define SHAPE5D(t) {\
    (int)t.size(0), (int)t.size(1), (int)t.size(2), (int)t.size(3), \
    (int)(t.size(1) * t.size(2) * t.size(3)),\
    (t.size(1) == 1) ? 0 : (int)(t.size(2) * t.size(3)),\
    (scalar_t*)t.data_ptr()\
}

torch::Tensor causalScan4d_cpu_Forward(
    torch::Tensor X, 
    torch::Tensor Z, 
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C
) {
    auto O = torch::zeros_like(X);
    AT_DISPATCH_FLOATING_TYPES(O.type(), "causalScan4d_cpu_Forward", ([&] {
        wrap_t<scalar_t> shapeX = SHAPE5D(X);
        wrap_t<scalar_t> shapeZ = SHAPE5D(Z);
        wrap_t<scalar_t> shapeA = SHAPE5D(A);
        wrap_t<scalar_t> shapeB = SHAPE5D(B);
        wrap_t<scalar_t> shapeC = SHAPE5D(C);
        wrap_t<scalar_t> shapeO = SHAPE5D(O);
        for(int ib=0; ib<shapeZ.b; ib++)
        for(int ih=0; ih<shapeZ.d; ih++)
        for(int in=0; in<shapeZ.n; in++)
        {
            INDICS indics[] = {
                {ib, ih},
                {in}
            };
            device::causalScan4d_cpu_Forward<scalar_t>(
                shapeX,
                shapeZ,
                shapeA,
                shapeB,
                shapeC,
                shapeO,
                indics[0],
                indics[1]
            );
        }
    }));
    return O;
}

std::vector<torch::Tensor> causalScan4d_cpu_Backward(
    torch::Tensor gradO,
    torch::Tensor X, 
    torch::Tensor Z,
    torch::Tensor A,
    torch::Tensor B, 
    torch::Tensor C
) {
    auto gradX = torch::zeros_like(X);
    auto gradZ = torch::zeros_like(Z);
    auto gradA = torch::zeros_like(A);
    auto gradB = torch::zeros_like(B);
    auto gradC = torch::zeros_like(C);
    AT_DISPATCH_FLOATING_TYPES(gradO.type(), "causalScan4d_cpu_Backward", ([&] {
        wrap_t<scalar_t> deltaX = SHAPE5D(gradX);
        wrap_t<scalar_t> deltaO = SHAPE5D(gradO);
        wrap_t<scalar_t> deltaZ = SHAPE5D(gradZ);
        wrap_t<scalar_t> deltaA = SHAPE5D(gradA);
        wrap_t<scalar_t> deltaB = SHAPE5D(gradB);
        wrap_t<scalar_t> deltaC = SHAPE5D(gradC);
        for(int ib=0; ib<deltaZ.b; ib++)
        for(int ih=0; ih<deltaZ.d; ih++)
        for(int in=0; in<deltaZ.n; in++)
        {
            INDICS indics[] = {
                {ib, ih},
                {in}
            };
            device::causalScan4d_cpu_Backward<scalar_t>(
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
                deltaC,
                indics[0],
                indics[1]
            );
        }
    }));
    return {gradX, gradZ, gradA, gradB, gradC};
}

#ifndef __PYBINDED__
#define __PYBINDED__
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &causalScan4d_cpu_Forward, "");
    m.def("backward", &causalScan4d_cpu_Backward, "");
}
#endif//__PYBINDED__