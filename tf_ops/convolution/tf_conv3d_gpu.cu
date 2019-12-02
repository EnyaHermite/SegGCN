// nnIndex: B*M*K;
// nnCount: B*M;
// input:   B*N*C;
// filter:  filter_size*C*r;
// output:  B*M*(C*r)
__global__ void depthwise_conv3d_forward(int B, int N, int M, int C, int r, int K, const int* nnIndex,
                                         const int* nnCount, const int* binIndex, const float* input,
                                         const float* filter, float* output)
{
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=blockIdx.y*blockDim.x+threadIdx.x;j<M*(C*r);j+=blockDim.x*gridDim.y)
        {
            int cout = j%(C*r); // output channel ID
            int cin = cout/r;   // input channel ID
            int m = j/(C*r);    // output point ID
            int nnSize = nnCount[i*M+m];

            for(int k=0;k<nnSize;k++)
            {
                int n = nnIndex[i*M*K+m*K+k];   // input point ID
                int f = binIndex[i*M*K+m*K+k];

                output[i*M*C*r+j] += input[i*N*C+n*C+cin]*filter[f*C*r+cout]/nnSize;
            }
        }
    }
}


__global__ void depthwise_input_backward(int B, int N, int M, int C, int r, int K, const int* nnIndex,
                                         const int* nnCount, const int* binIndex, const float* input,
                                         const float* filter, const float* gradOutput, float* gradInput)
{
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=blockIdx.y*blockDim.x+threadIdx.x;j<M*(C*r);j+=blockDim.x*gridDim.y)
        {
            int cout = j%(C*r); // output channel ID
            int cin = cout/r;   // input channel ID
            int m = j/(C*r);    // output point ID
            int nnSize = nnCount[i*M+m];

            for(int k=0;k<nnSize;k++)
            {
                int n = nnIndex[i*M*K+m*K+k];   // input point ID
                int f = binIndex[i*M*K+m*K+k];

                float derIn = gradOutput[i*M*C*r+j]*filter[f*C*r+cout]/nnSize;
                atomicAdd(&gradInput[i*N*C+n*C+cin],derIn);
            }
        }
    }
}


__global__ void depthwise_filter_backward(int B, int N, int M, int C, int r, int K, const int* nnIndex,
                                          const int* nnCount, const int* binIndex, const float* input,
                                          const float* gradOutput, float* gradFilter, int sharedMemSize,
                                          int startIdx)
{
    extern __shared__ float gradPerBlock[]; // the gradient on each block
    for (int i=threadIdx.x;i<sharedMemSize;i+=blockDim.x)
    {
        gradPerBlock[i] = 0; // for 1D block
    }
    __syncthreads();

    int endIdx = sharedMemSize+startIdx;
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=blockIdx.y*blockDim.x+threadIdx.x;j<M*(C*r);j+=blockDim.x*gridDim.y)
        {
            int cout = j%(C*r); // output channel ID
            int cin = cout/r;   // input channel ID
            int m = j/(C*r);    // output point ID
            int nnSize = nnCount[i*M+m];

            for(int k=0;k<nnSize;k++)
            {
                int n = nnIndex[i*M*K+m*K+k];   // input point ID
                int f = binIndex[i*M*K+m*K+k];

                float derFilt = gradOutput[i*M*C*r+j]*input[i*N*C+n*C+cin]/nnSize;

                int currIdx = f*C*r+cout;
                if((currIdx>=startIdx) && (currIdx<endIdx)) // within the shared memory
                {
                    atomicAdd(&gradPerBlock[currIdx-startIdx],derFilt);
                }
            }
        }
    }
    __syncthreads();

    for (int i=threadIdx.x;i<sharedMemSize;i+=blockDim.x)
    {
        atomicAdd(&gradFilter[i+startIdx],gradPerBlock[i]); // for 1D block
    }
}


// nnIndex: B*M*K;
// nnCount: B*M;
// input:   B*N*C;
// filter:  filter_size*C*r;
// output:  B*M*(C*r)
__global__ void fuzzy_depthwise_conv3d_forward(int B, int N, int M, int C, int r, int K, const int* nnIndex,
                                               const int* nnCount, const int* binIndex, const float* binCoeff,
                                               const float* input, const float* filter, float* output)
{
    // T is the number of fuzzy bins each neighbor locates in
    const int T = 4;

    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=blockIdx.y*blockDim.x+threadIdx.x;j<M*(C*r);j+=blockDim.x*gridDim.y)
        {
            int cout = j%(C*r); // output channel ID
            int cin = cout/r;   // input channel ID
            int m = j/(C*r);    // output point ID
            int nnSize = nnCount[i*M+m];

            for(int k=0;k<nnSize;k++)
            {
                int n = nnIndex[i*M*K+m*K+k];   // input point ID

                float weight = 0;
                for(int idx=0;idx<T;idx++)
                {
                    int f = binIndex[i*M*K*T+m*K*T+k*T+idx];
                    float coeff = binCoeff[i*M*K*T+m*K*T+k*T+idx];

                    if (coeff>0)
                    {
                        weight += coeff*filter[f*C*r+cout];
                    }
                }

                output[i*M*C*r+j] += input[i*N*C+n*C+cin]*weight/nnSize;
            }
        }
    }
}


__global__ void fuzzy_depthwise_input_backward(int B, int N, int M, int C, int r, int K,
                                               const int* nnIndex, const int* nnCount, const int* binIndex,
                                               const float* binCoeff, const float* input, const float* filter,
                                               const float* gradOutput, float* gradInput)
{
    const int T = 4;

    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=blockIdx.y*blockDim.x+threadIdx.x;j<M*(C*r);j+=blockDim.x*gridDim.y)
        {
            int cout = j%(C*r); // output channel ID
            int cin = cout/r;   // input channel ID
            int m = j/(C*r);    // output point ID
            int nnSize = nnCount[i*M+m];

            for(int k=0;k<nnSize;k++)
            {
                int n = nnIndex[i*M*K+m*K+k];   // input point ID

                float weight = 0;
                for(int idx=0;idx<T;idx++)
                {
                    int f = binIndex[i*M*K*T+m*K*T+k*T+idx];
                    float coeff = binCoeff[i*M*K*T+m*K*T+k*T+idx];

                    if (coeff>0)
                    {
                        weight += coeff*filter[f*C*r+cout];
                    }
                }

                float derIn = gradOutput[i*M*C*r+j]*weight/nnSize;
                atomicAdd(&gradInput[i*N*C+n*C+cin],derIn);
            }
        }
    }
}


__global__ void fuzzy_depthwise_filter_backward(int B, int N, int M, int C, int r, int K, const int* nnIndex,
                                                const int* nnCount, const int* binIndex, const float* binCoeff,
                                                const float* input, const float* gradOutput, float* gradFilter,
                                                int sharedMemSize, int startIdx)
{
    const int T = 4;

    extern __shared__ float gradPerBlock[]; // the gradient on each block
    for (int i=threadIdx.x;i<sharedMemSize;i+=blockDim.x)
    {
        gradPerBlock[i] = 0; // for 1D block
    }
    __syncthreads();

    int endIdx = sharedMemSize+startIdx;
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=blockIdx.y*blockDim.x+threadIdx.x;j<M*(C*r);j+=blockDim.x*gridDim.y)
        {
            int cout = j%(C*r); // output channel ID
            int cin = cout/r;   // input channel ID
            int m = j/(C*r);    // output point ID
            int nnSize = nnCount[i*M+m];

            for(int k=0;k<nnSize;k++)
            {
                int n = nnIndex[i*M*K+m*K+k];   // input point ID
                float derFilt = gradOutput[i*M*C*r+j]*input[i*N*C+n*C+cin]/nnSize;

                for(int idx=0;idx<T;idx++)
                {
                    int f = binIndex[i*M*K*T+m*K*T+k*T+idx];
                    float coeff = binCoeff[i*M*K*T+m*K*T+k*T+idx];

                    if (coeff>0)
                    {
                        int currIdx = f*C*r+cout;
                        if((currIdx>=startIdx) && (currIdx<endIdx)) // within the shared memory
                        {
                            atomicAdd(&gradPerBlock[currIdx-startIdx],coeff*derFilt);
                        }
                    }
                }
            }
        }
    }
    __syncthreads();

    for (int i=threadIdx.x;i<sharedMemSize;i+=blockDim.x)
    {
        atomicAdd(&gradFilter[i+startIdx],gradPerBlock[i]); // for 1D block
    }
}




void depthwiseConv3dLauncher(int B, int N, int M, int C, int r, int K, const int* nnIndex,
                             const int* nnCount, const int* binIndex, const float* input,
                             const float* filter, float* output)
{
    depthwise_conv3d_forward<<<B,1024>>>(B, N, M, C, r, K, nnIndex, nnCount, binIndex,
                                          input, filter, output);
}

void depthwiseConv3dGradLauncher(int B, int N, int M, int F, int C, int r, int K,
                                 const int* nnIndex, const int* nnCount, const int* binIndex,
                                 const float* input,  const float* filter, const float* gradOutput,
                                 float* gradInput, float* gradFilter)
{
    // titan xp has shared memory of 49152 bytes, each float value takes 4 bytes in the memory
    int maxSharedMemSize = int(49152/sizeof(float));

    depthwise_input_backward<<<B,1024>>>(B, N, M, C, r, K, nnIndex, nnCount, binIndex,
                                          input, filter, gradOutput, gradInput);

    int maxIter = (F*C*r)/maxSharedMemSize;
    int remainder = (F*C*r)%maxSharedMemSize;
    for(int iter=0;iter<maxIter;iter++)
    {
        depthwise_filter_backward<<<B,1024,sizeof(float)*maxSharedMemSize>>>(B, N, M, C, r, K, nnIndex, nnCount,
                                                                             binIndex, input, gradOutput, gradFilter,
                                                                             maxSharedMemSize, maxSharedMemSize*iter);
    }
    if(remainder>0) // fill the remainder
    {
        depthwise_filter_backward<<<B,1024,sizeof(float)*remainder>>>(B, N, M, C, r, K, nnIndex, nnCount,
                                                                      binIndex, input, gradOutput, gradFilter,
                                                                      remainder, maxSharedMemSize*maxIter);
    }
}


void fuzzyDepthwiseConv3dLauncher(int B, int N, int M, int C, int r, int K, const int* nnIndex,
                                  const int* nnCount, const int* binIndex, const float* binCoeff,
                                  const float* input, const float* filter, float* output)
{
    fuzzy_depthwise_conv3d_forward<<<B,1024>>>(B, N, M, C, r, K, nnIndex, nnCount, binIndex,
                                               binCoeff, input, filter, output);
}

void fuzzyDepthwiseConv3dGradLauncher(int B, int N, int M, int F, int C, int r, int K, const int* nnIndex,
                                      const int* nnCount, const int* binIndex, const float* binCoeff,
                                      const float* input,  const float* filter, const float* gradOutput,
                                      float* gradInput, float* gradFilter)
{
    // titan xp has shared memory of 49152 bytes, each float value takes 4 bytes in the memory
    int maxSharedMemSize = int(49152/sizeof(float));

    fuzzy_depthwise_input_backward<<<B,1024>>>(B, N, M, C, r, K, nnIndex, nnCount, binIndex, binCoeff,
                                               input, filter, gradOutput, gradInput);

    int maxIter = (F*C*r)/maxSharedMemSize;
    int remainder = (F*C*r)%maxSharedMemSize;
    for(int iter=0;iter<maxIter;iter++)
    {
        fuzzy_depthwise_filter_backward<<<B,1024,sizeof(float)*maxSharedMemSize>>>(B, N, M, C, r, K, nnIndex, nnCount,
                                                                                   binIndex, binCoeff, input, gradOutput,
                                                                                   gradFilter, maxSharedMemSize,
                                                                                   maxSharedMemSize*iter);
    }
    if(remainder>0) // fill the remainder
    {
        fuzzy_depthwise_filter_backward<<<B,1024,sizeof(float)*remainder>>>(B, N, M, C, r, K, nnIndex, nnCount,
                                                                            binIndex, binCoeff, input, gradOutput,
                                                                            gradFilter, remainder,
                                                                            maxSharedMemSize*maxIter);
    }
}
