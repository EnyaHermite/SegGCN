// nnCount:   B*M
// nnDist:    B*M*nnSample
// Weight:    B*M*nnSample
__global__ void cal_weight(int B, int M, int nnSample, int weightType, float radius,
                           const int* nnCount, const float* nnDist, float* Weight)
{
    // get the neighbor indices
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=threadIdx.x;j<M;j+=blockDim.x)
        {
            int K = nnCount[i*M+j];

            for(int k=0;k<K;k++)
            {
                float dist = max(nnDist[i*M*nnSample+j*nnSample+k],1e-15);

                if (weightType==0)
                {
                    Weight[i*M*nnSample+j*nnSample+k] = float(1)/dist; // inverse sqrt distance
                }
                else
                {
                    Weight[i*M*nnSample+j*nnSample+k] = max(0.0, 1 - dist/radius); // bilinear like
                }
            }
        }
    }
}


void buildSampleWeightLauncher(int B, int M, int nnSample, int weightType, float radius,
                               const int* nnCount, const float* nnDist, float* Weight)
{
    cal_weight<<<B,1024>>>(B, M, nnSample, weightType, radius, nnCount, nnDist, Weight);
}
