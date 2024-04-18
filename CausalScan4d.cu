#ifndef __COMMON_H__
#define __COMMON_H__
    template <typename scalar_t> struct wrap_t{
        int b, l, d, n;
        int stepb, stepl;
        scalar_t* p;
    };
    typedef struct{
        int x;
    }INDICS;
    #define SHIFT_BLOCK_SIZE 8
    #define BLOCK_SIZE (1<<SHIFT_BLOCK_SIZE)
    #define SHAPE5D(t) {\
        (int)t.size(0), (int)t.size(1), (int)t.size(2), (int)t.size(3), \
        (int)(t.size(1) * t.size(2) * t.size(3)),\
        t.size(1) == 1 ? 0 : (int)(t.size(2) * t.size(3)),\
        (scalar_t*)t.data_ptr()\
    }
    #define SHAPE5D_Z(t) {\
        (int)t.size(0), (int)t.size(1), (int)t.size(2), (int)t.size(3), \
        (int)(t.size(1) * t.size(2) * t.size(3)),\
        (int)(t.size(2) * t.size(3)),\
        (scalar_t*)t.data_ptr()\
    }
    #define IDX5D(shape) ((ib % shape.b) * shape.stepb + (id % shape.d) * shape.n + in % shape.n)
    #define Ptr5D(shape) (shape.p + IDX5D(shape))
    #define GROUP_SIZE 1023
#endif//__COMMON_H__

#ifndef __DISABLE_CUDA__
    #define DEVICEINDICS 
    #define CAUSAL_FORWARD causalScan4d_Forward_cuda
    #define CAUSAL_BACKWARD causalScan4d_Backward_cuda
    #define atomAdd atomicAdd
    #include <cuda.h>
    #include <cuda_runtime.h>
#else//__DISABLE_CUDA__
    #ifdef DEVICEINDICS
        #undef DEVICEINDICS
    #endif//
    #ifdef CAUSAL_FORWARD
        #undef CAUSAL_FORWARD
    #endif//
    #ifdef CAUSAL_BACKWARD
        #undef CAUSAL_BACKWARD
    #endif//
    #ifdef __global__
        #undef __global__
    #endif//
    #ifdef atomAdd
        #undef atomAdd
    #endif//
    #define DEVICEINDICS ,const INDICS& blockIdx, const INDICS& threadIdx
    #define CAUSAL_FORWARD causalScan4d_Forward_cpu
    #define CAUSAL_BACKWARD causalScan4d_Backward_cpu
    #define __global__
    #define atomAdd(p,b) (*(p) = *(p) + (b))
#endif//__DISABLE_CUDA__

namespace { namespace device {
    template <typename scalar_t> __global__ void CAUSAL_FORWARD(
        const wrap_t<scalar_t> shapeX,
        const wrap_t<scalar_t> shapeZ,
        const wrap_t<scalar_t> shapeS,
        const wrap_t<scalar_t> shapeA,
        const wrap_t<scalar_t> shapeB,
        const wrap_t<scalar_t> shapeC,
        const wrap_t<scalar_t> shapeO,
        int range
        DEVICEINDICS
    )
    {
        int idx = blockIdx.x << SHIFT_BLOCK_SIZE | threadIdx.x;
        if( idx >= range ) return;
        int ib = idx / shapeS.stepl;
        int idn = idx % shapeS.stepl;
        int id = idn / shapeS.n;
        int in = idn % shapeS.n;

        int sx = IDX5D(shapeX);
        scalar_t * pX = shapeX.p + sx;
        scalar_t * pO = shapeO.p + sx;
        scalar_t * pZ = shapeZ.p + ib * shapeZ.stepb + idn;
        scalar_t * pS = shapeS.p + ib * shapeS.stepb + idn;
        scalar_t * pA = Ptr5D(shapeA);
        scalar_t * pB = Ptr5D(shapeB);
        scalar_t * pC = Ptr5D(shapeC);
        scalar_t zh = *pZ;
        int i = 0;
        while(i<shapeO.l) {
            if( i % GROUP_SIZE == 0 ) {
                *pS = zh;
                pS += shapeS.stepl;
            }
            zh = (*pA) * zh + (*pB) * (*pX);
            atomAdd(pO, ((*pC) * zh));
            pX += shapeX.stepl;
            pA += shapeA.stepl;
            pB += shapeB.stepl;
            pC += shapeC.stepl;
            pO += shapeO.stepl;
            i++;
        }
        *pZ = zh;
    }

    template <typename scalar_t> __global__ void CAUSAL_BACKWARD(
        const wrap_t<scalar_t> shapeX,
        const wrap_t<scalar_t> shapeZ,
        const wrap_t<scalar_t> shapeS,
        const wrap_t<scalar_t> shapeA,
        const wrap_t<scalar_t> shapeB,
        const wrap_t<scalar_t> shapeC,
        scalar_t * pGradO,
        scalar_t * pGradX,
        scalar_t * pGradZ,
        scalar_t * pGradA,
        scalar_t * pGradB,
        scalar_t * pGradC,
        int range
        DEVICEINDICS
    )
    {
        int idx = blockIdx.x << SHIFT_BLOCK_SIZE | threadIdx.x;
        if( idx >= range ) return;
        int ib = idx / shapeS.stepl;
        int idn = idx % shapeS.stepl;
        int id = idn / shapeS.n;
        int in = idn % shapeS.n;

        int sx = IDX5D(shapeX);
        int sz = ib * shapeZ.stepb + idn;
        int ss = ib * shapeS.stepb + idn;
        int sa = IDX5D(shapeA);
        int sb = IDX5D(shapeB);
        int sc = IDX5D(shapeC);
        scalar_t * pX = shapeX.p + sx;
        scalar_t * pS = shapeS.p + ss;
        scalar_t * pA = shapeA.p + sa;
        scalar_t * pB = shapeB.p + sb;
        scalar_t * pC = shapeC.p + sc;
        pGradO += sx;
        pGradX += sx;
        pGradZ += sz;
        pGradA += sa;
        pGradB += sb;
        pGradC += sc;

        int length = shapeX.l;
        scalar_t gradh = 0.0;
        scalar_t zhs[GROUP_SIZE+1];
        int groups = (length + GROUP_SIZE - 1) / GROUP_SIZE;
        for(int igroups=groups-1; igroups>=0; igroups--){
            int ibegin = igroups * GROUP_SIZE;
            int group_length = (igroups==groups-1)?(length-ibegin):GROUP_SIZE;

            scalar_t * pIX = pX + ibegin*shapeX.stepl;
            scalar_t * pIA = pA + ibegin*shapeA.stepl;
            scalar_t * pIB = pB + ibegin*shapeB.stepl;
            zhs[0] = (*pS);
            pS += shapeS.stepl;
            for(int i=0; i<group_length; i++) {
                zhs[i+1] = (*pIA) * zhs[i] + (*pIB) * (*pIX);
                pIA += shapeA.stepl;
                pIB += shapeB.stepl;
                pIX += shapeX.stepl;
            }

            int iend = ibegin + group_length;
            scalar_t * pIC = pC + iend * shapeC.stepl;
            scalar_t * pIGradO = pGradO + iend * shapeX.stepl;
            scalar_t * pIGradX = pGradX + iend * shapeX.stepl;
            scalar_t * pIGradA = pGradA + iend * shapeA.stepl;
            scalar_t * pIGradB = pGradB + iend * shapeB.stepl;
            scalar_t * pIGradC = pGradC + iend * shapeC.stepl;
            while(group_length-->0) {
                pIA -= shapeA.stepl;
                pIB -= shapeB.stepl;
                pIX -= shapeX.stepl;
                pIC -= shapeC.stepl;
                pIGradO -= shapeX.stepl;
                pIGradX -= shapeX.stepl;
                pIGradA -= shapeA.stepl;
                pIGradB -= shapeB.stepl;
                pIGradC -= shapeC.stepl;

                atomAdd(pIGradC, (*pIGradO) * zhs[group_length+1]);
                gradh += (*pIGradO) * (*pIC);
                atomAdd(pIGradB, gradh * (*pIX));
                atomAdd(pIGradX, gradh * (*pIB));
                atomAdd(pIGradA, zhs[group_length] * gradh);
                gradh *= (*pIA);
            }
        }
        *pGradZ = gradh;
    }
}}

#ifndef __TORCH_INLINE__
#define __TORCH_INLINE__

#ifndef __DISABLE_CUDA__
#define __DISABLE_CUDA__
#include "CausalScan4d.cu"
#undef __DISABLE_CUDA__
#endif//__DISABLE_CUDA__

#include <torch/extension.h>
#include <vector>
torch::Tensor causalScan4d_Forward(
    torch::Tensor X, 
    torch::Tensor Z, 
    torch::Tensor S,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C
) {
    auto O = torch::zeros_like(X);
    int range = (int)(Z.size(0) * Z.size(2) * Z.size(3));
    if(X.is_cuda()){
        #ifndef __DISABLE_CUDA__
        AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "causalScan4d_Forward", ([&] {
            wrap_t<scalar_t> shapeX = SHAPE5D_Z(X);
            wrap_t<scalar_t> shapeZ = SHAPE5D_Z(Z);
            wrap_t<scalar_t> shapeS = SHAPE5D_Z(S);
            wrap_t<scalar_t> shapeA = SHAPE5D(A);
            wrap_t<scalar_t> shapeB = SHAPE5D(B);
            wrap_t<scalar_t> shapeC = SHAPE5D(C);
            wrap_t<scalar_t> shapeO = SHAPE5D_Z(O);
            int blocks = (range + BLOCK_SIZE - 1) / BLOCK_SIZE;
            device::causalScan4d_Forward_cuda<scalar_t><<<blocks, BLOCK_SIZE>>>(
                shapeX, shapeZ, shapeS, shapeA, shapeB, shapeC, shapeO,
                range
            );
        }));
        #else
        AT_ASSERT(false);
        #endif//__DISABLE_CUDA__
    }else{
        AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "causalScan4d_Forward", ([&] {
            wrap_t<scalar_t> shapeX = SHAPE5D_Z(X);
            wrap_t<scalar_t> shapeZ = SHAPE5D_Z(Z);
            wrap_t<scalar_t> shapeS = SHAPE5D_Z(S);
            wrap_t<scalar_t> shapeA = SHAPE5D(A);
            wrap_t<scalar_t> shapeB = SHAPE5D(B);
            wrap_t<scalar_t> shapeC = SHAPE5D(C);
            wrap_t<scalar_t> shapeO = SHAPE5D_Z(O);
            at::parallel_for(0, shapeZ.b*shapeZ.d, 0, [&](int64_t start, int64_t end){
                while(start<end){
                    for(int in=0; in<shapeZ.n; in++)
                    {
                        int pos = start * shapeZ.n + in;
                        INDICS indics[] = {
                            {(int)(pos >> SHIFT_BLOCK_SIZE )},
                            {(int)(pos % BLOCK_SIZE)}
                        };
                        device::causalScan4d_Forward_cpu<scalar_t>(
                            shapeX, shapeZ, shapeS, shapeA, shapeB, shapeC, shapeO,
                            range,
                            indics[0], indics[1]
                        );
                    }
                    start++;
                };
            });
        }));
    }
    return O;
}

std::vector<torch::Tensor> causalScan4d_Backward(
    torch::Tensor gradO,
    torch::Tensor gradZ,
    torch::Tensor X, 
    torch::Tensor S,
    torch::Tensor A,
    torch::Tensor B, 
    torch::Tensor C
) {
    auto gradX = torch::zeros_like(X);
    auto gradA = torch::zeros_like(A);
    auto gradB = torch::zeros_like(B);
    auto gradC = torch::zeros_like(C);
    int range = (int)(S.size(0) * S.size(2) * S.size(3));
    if(gradO.is_cuda()) {
        #ifndef __DISABLE_CUDA__
        AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "causalScan4d_Backward", ([&] {
            wrap_t<scalar_t> shapeX = SHAPE5D_Z(X);
            wrap_t<scalar_t> shapeZ = SHAPE5D_Z(gradZ);
            wrap_t<scalar_t> shapeS = SHAPE5D_Z(S);
            wrap_t<scalar_t> shapeA = SHAPE5D(A);
            wrap_t<scalar_t> shapeB = SHAPE5D(B);
            wrap_t<scalar_t> shapeC = SHAPE5D(C);
            int blocks = (range + BLOCK_SIZE - 1) >> SHIFT_BLOCK_SIZE;
            device::causalScan4d_Backward_cuda<scalar_t><<<blocks, BLOCK_SIZE>>>(
                shapeX, shapeZ, shapeS, shapeA, shapeB, shapeC, 
                (scalar_t*)gradO.data_ptr(),
                (scalar_t*)gradX.data_ptr(),
                (scalar_t*)gradZ.data_ptr(),
                (scalar_t*)gradA.data_ptr(),
                (scalar_t*)gradB.data_ptr(),
                (scalar_t*)gradC.data_ptr(),
                range
            );
        }));
        #else
        AT_ASSERT(false);
        #endif//__DISABLE_CUDA__
    }else{
        AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "causalScan4d_Backward", ([&] {
            wrap_t<scalar_t> shapeX = SHAPE5D_Z(X);
            wrap_t<scalar_t> shapeZ = SHAPE5D_Z(gradZ);
            wrap_t<scalar_t> shapeS = SHAPE5D_Z(S);
            wrap_t<scalar_t> shapeA = SHAPE5D(A);
            wrap_t<scalar_t> shapeB = SHAPE5D(B);
            wrap_t<scalar_t> shapeC = SHAPE5D(C);
            for(int start=0; start<range; start++){
                INDICS indics[] = {
                    {(int)(start >> SHIFT_BLOCK_SIZE)},
                    {(int)(start % BLOCK_SIZE)}
                };
                device::causalScan4d_Backward_cpu<scalar_t>(
                    shapeX, shapeZ, shapeS, shapeA, shapeB, shapeC, 
                    (scalar_t*)gradO.data_ptr(),
                    (scalar_t*)gradX.data_ptr(),
                    (scalar_t*)gradZ.data_ptr(),
                    (scalar_t*)gradA.data_ptr(),
                    (scalar_t*)gradB.data_ptr(),
                    (scalar_t*)gradC.data_ptr(),
                    range,
                    indics[0], indics[1]
                );
            }
        }));
    }
    return {gradX, gradZ, gradA, gradB, gradC};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &causalScan4d_Forward, "");
    m.def("backward", &causalScan4d_Backward, "");
}
#endif//__TORCH_INLINE__