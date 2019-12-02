#ifndef M_PI
#define M_PI           3.141592653589793F  /* pi */
#endif

#ifndef M_EPS
#define M_EPS          1.01e-3F             /* epsilon */
#endif

struct point3d
{
    float x=0, y=0, z=0;
};

// database:  B*N*3, (x,y,z)
// query:     B*M*3, (x,y,z)
// nnIndex:   B*M*K
// nnCount:   B*M
// nnDist:    B*M*K
// filtIndex: B*M*K
__global__ void build_spherical_kernel(int B, int N, int M, int K, int n, int p, int q,
                                       const float radius, const float* database,
                                       const float* query, const int* nnIndex,
                                       const int* nnCount, const float* nnDist, int* filtIndex)
{
    // get the neighbor indices
    point3d ptQuery, pt, delta;
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=threadIdx.x;j<M;j+=blockDim.x)
        {
            ptQuery.x = query[i*M*3+j*3];
            ptQuery.y = query[i*M*3+j*3+1];
            ptQuery.z = query[i*M*3+j*3+2];

            int nnSize = nnCount[i*M+j];
            for(int k=0;k<nnSize;k++)
            {
                int ptID = nnIndex[i*M*K+j*K+k];   // input point ID

                pt.x = database[i*N*3+ptID*3];
                pt.y = database[i*N*3+ptID*3+1];
                pt.z = database[i*N*3+ptID*3+2];

                delta.x = pt.x - ptQuery.x;
                delta.y = pt.y - ptQuery.y;
                delta.z = pt.z - ptQuery.z;

                float dist = nnDist[i*M*K+j*K+k]; // the sqrt distance
                float dist2D = delta.x*delta.x + delta.y*delta.y;
                dist2D = sqrtf(dist2D);

                filtIndex[i*M*K+j*K+k] = 0;
                if (dist>M_EPS) // update the bin index
                {
                    float theta = atan2f(delta.y, delta.x);
                    float phi = atan2f(delta.z, dist2D);

                    theta = theta<M_PI?theta:(-M_PI);
                    theta = theta>(-M_PI)?theta:(-M_PI);
                    theta += M_PI;

                    phi = phi<(M_PI/2)?phi:(M_PI/2);
                    phi = phi>(-M_PI/2)?phi:(-M_PI/2);
                    phi += M_PI/2;

                    float alpha = theta*n/2/M_PI;
                    float beta = phi*p/M_PI;
                    float gamma = dist*q/radius;

                    int nID = min(n-1, int(alpha));
                    int pID = min(p-1, int(beta));
                    int qID = min(q-1, int(gamma));

                    filtIndex[i*M*K+j*K+k] = qID*p*n + pID*n + nID + 1;
                }
            }
        }
    }
}

// database:  B*N*3, (x,y,z)
// query:     B*M*3, (x,y,z)
// nnIndex:   B*M*K
// nnCount:   B*M
// nnDist:    B*M*K
// filtIndex: B*M*K
__global__ void build_fuzzy_spherical_kernel(int B, int N, int M, int K, int n, int p, int q,
                                             const float radius, const float* database, const float* query,
                                             const int* nnIndex, const int* nnCount, const float* nnDist,
                                             int* filtIndex, float* filtCoeff)
{
    const int F = 4;

    // get the neighbor indices
    point3d ptQuery, pt, delta;
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=threadIdx.x;j<M;j+=blockDim.x)
        {
            ptQuery.x = query[i*M*3+j*3];
            ptQuery.y = query[i*M*3+j*3+1];
            ptQuery.z = query[i*M*3+j*3+2];

            int nnSize = nnCount[i*M+j];
            for(int k=0;k<nnSize;k++)
            {
                int ptID = nnIndex[i*M*K+j*K+k];   // input point ID

                pt.x = database[i*N*3+ptID*3];
                pt.y = database[i*N*3+ptID*3+1];
                pt.z = database[i*N*3+ptID*3+2];

                delta.x = pt.x - ptQuery.x;
                delta.y = pt.y - ptQuery.y;
                delta.z = pt.z - ptQuery.z;

                float dist = nnDist[i*M*K+j*K+k]; // the sqrt distance
                float dist2D = delta.x*delta.x + delta.y*delta.y;
                dist2D = sqrtf(dist2D);

                float selfCoeff = max(0.0,1-dist/radius);
                filtIndex[i*M*K*F+j*K*F+k*F] = 0;
                filtCoeff[i*M*K*F+j*K*F+k*F] = selfCoeff; // self-loop coeffcient

                // compute the coefficients of fuzzy bins
                float theta = atan2f(delta.y, delta.x);
                float phi = atan2f(delta.z, dist2D);

                theta = theta<M_PI?theta:(-M_PI);
                theta = theta>(-M_PI)?theta:(-M_PI);
                theta += M_PI;

                phi = phi<(M_PI/2)?phi:(M_PI/2);
                phi = phi>(-M_PI/2)?phi:(-M_PI/2);
                phi += M_PI/2;

                int nID, pID[2], qID;
                float alpha, beta[2], gamma;

                alpha   = theta*n/2/M_PI;
                beta[0] = phi*p/M_PI;
                gamma   = dist*q/radius;

                nID    = min(n-1, int(alpha));
                pID[0] = min(p-1, int(beta[0]));
                qID    = min(q-1, int(gamma));

                beta[0]  -= pID[0];

                pID[1] = beta[0]<0.5? max(0,pID[0]-1):min(p-1,pID[0]+1);

                int pIN = (pID[0]==pID[1])?1:2;
                beta[1]  = (pIN==1)?0:abs(beta[0] - 0.5);
                beta[0]  = 1 - beta[1];

                for(int pi=0;pi<pIN;pi++)
                {
                    int f = qID*p*n + pID[pi]*n + nID + 1;
                    filtIndex[i*M*K*F+j*K*F+k*F+pi+1] = f;
                    filtCoeff[i*M*K*F+j*K*F+k*F+pi+1] = beta[pi]*(1-selfCoeff);
                }
            }
        }
    }
}


void sphericalKernelLauncher(int B, int N, int M, int K, int n, int p, int q, float radius,
                             const float* database, const float* query, const int* nnIndex,
                             const int* nnCount, const float* nnDist, int* filtIndex)
{
    build_spherical_kernel<<<B,1024>>>(B, N, M, K, n, p, q, radius,
                                database, query, nnIndex, nnCount, nnDist, filtIndex);
}

void fuzzySphericalKernelLauncher(int B, int N, int M, int K, int n, int p, int q, float radius,
                                  const float* database, const float* query, const int* nnIndex,
                                  const int* nnCount, const float* nnDist,
                                  int* filtIndex, float* filtCoeff)
{
    build_fuzzy_spherical_kernel<<<B,1024>>>(B, N, M, K, n, p, q, radius,
                                database, query, nnIndex, nnCount, nnDist, filtIndex, filtCoeff);
}








__global__ void build_kpconv_kernel(int B, int N, int M, int T, int K, float sigma,
                                    const float* database, const float* query, const float* kernel,
                                    const int* nnIndex, const int* nnCount, int* filtIndex)
{
    // get the neighbor indices
    point3d ptQuery, pt, delta;
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=threadIdx.x;j<M;j+=blockDim.x)
        {
            ptQuery.x = query[i*M*3+j*3];
            ptQuery.y = query[i*M*3+j*3+1];
            ptQuery.z = query[i*M*3+j*3+2];

            int nnSize = nnCount[i*M+j];
            for(int k=0;k<nnSize;k++)
            {
                int ptID = nnIndex[i*M*K+j*K+k];   // input point ID

                pt.x = database[i*N*3+ptID*3];
                pt.y = database[i*N*3+ptID*3+1];
                pt.z = database[i*N*3+ptID*3+2];

                delta.x = pt.x - ptQuery.x;
                delta.y = pt.y - ptQuery.y;
                delta.z = pt.z - ptQuery.z;

                float min_dist = 1e20;
                int   binIndex = 0;
                for(int t=0;t<T;t++)
                {
                    float dist = (delta.x-kernel[t*3])  *(delta.x-kernel[t*3])   +
                                 (delta.y-kernel[t*3+1])*(delta.y-kernel[t*3+1]) +
                                 (delta.z-kernel[t*3+2])*(delta.z-kernel[t*3+2]);
                    if (dist<min_dist)
                    {
                        min_dist = dist;
                        binIndex = t;
                    }
                }

                filtIndex[i*M*K+j*K+k] = binIndex;
            }
        }
    }
}

// database:  B*N*3, (x,y,z)
// query:     B*M*3, (x,y,z)
// nnIndex:   B*M*K
// nnCount:   B*M
// nnDist:    B*M*K
// filtIndex: B*M*K
__global__ void build_fuzzy_kpconv_kernel(int B, int N, int M, int T, int K, float sigma,
                                          const float* database, const float* query, const float* kernel,
                                          const int* nnIndex, const int* nnCount, int* filtIndex, float* filtCoeff)
{
    const int F = 4;

    // get the neighbor indices
    point3d ptQuery, pt, delta;
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=threadIdx.x;j<M;j+=blockDim.x)
        {
            ptQuery.x = query[i*M*3+j*3];
            ptQuery.y = query[i*M*3+j*3+1];
            ptQuery.z = query[i*M*3+j*3+2];

            int nnSize = nnCount[i*M+j];
            for(int k=0;k<nnSize;k++)
            {
                int ptID = nnIndex[i*M*K+j*K+k];   // input point ID

                pt.x = database[i*N*3+ptID*3];
                pt.y = database[i*N*3+ptID*3+1];
                pt.z = database[i*N*3+ptID*3+2];

                delta.x = pt.x - ptQuery.x;
                delta.y = pt.y - ptQuery.y;
                delta.z = pt.z - ptQuery.z;

                int iter = 0;
                float coeff;
                for(int t=0;t<T;t++)
                {
                    float dist = (delta.x-kernel[t*3])*(delta.x-kernel[t*3])     +
                                 (delta.y-kernel[t*3+1])*(delta.y-kernel[t*3+1]) +
                                 (delta.z-kernel[t*3+2])*(delta.z-kernel[t*3+2]);
                    coeff = max(0.0,1-sqrtf(dist)/sigma);

                    if (coeff>0)
                    {
                        filtIndex[i*M*K*F+j*K*F+k*F+iter] = t;
                        filtCoeff[i*M*K*F+j*K*F+k*F+iter] = coeff;
                        iter++;
                    }
                }
            }
        }
    }
}

void kpconvKernelLauncher(int B, int N, int M, int T, int K, float sigma, const float* database,
                          const float* query, const float* kernel, const int* nnIndex,
                          const int* nnCount, int* filtIndex)
{
    build_kpconv_kernel<<<B,1024>>>(B, N, M, T, K, sigma, database, query, kernel,
                                    nnIndex, nnCount, filtIndex);
}

void fuzzyKpconvKernelLauncher(int B, int N, int M, int T, int K, float sigma, const float* database,
                               const float* query, const float* kernel, const int* nnIndex,
                               const int* nnCount, int* filtIndex, float* filtCoeff)
{
    build_fuzzy_kpconv_kernel<<<B,1024>>>(B, N, M, T, K, sigma, database, query, kernel, nnIndex,
                                          nnCount, filtIndex, filtCoeff);
}