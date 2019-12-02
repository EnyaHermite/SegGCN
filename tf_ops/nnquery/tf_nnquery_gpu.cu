struct point3d
{
    float x=0, y=0, z=0;
};

// database:  B*N*3
// query:     B*M*3
// nnIndex:   B*M*nnSample
// nnCount:   B*M
// nnDist:    B*M*nnSample
__global__ void cal_nnidx_sphere(int B, int N, int M, int nnSample, float radius,
                                 const float* database, const float* query,
                                 int* nnIndex, int* nnCount, float* nnDist)
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

            int s=0; // to count the number of neighbors
            while(s==0) //require a minimum of 1 neighbor point
            {
                //re-initialziation
                s = 0;

                for(int k=0;k<N;k++)
                {
                    pt.x = database[i*N*3+k*3];
                    pt.y = database[i*N*3+k*3+1];
                    pt.z = database[i*N*3+k*3+2];

                    delta.x = pt.x - ptQuery.x;
                    delta.y = pt.y - ptQuery.y;
                    delta.z = pt.z - ptQuery.z;

                    float dist3D = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z; // squared 3D
                    dist3D = sqrtf(dist3D); //sqrt

                    if (dist3D<radius) // find a neighbor in range
                    {
                        if (s<nnSample) // sample NO=nnSample neighbor points, requires shuffling of points order in every epoch
                        {
                            nnIndex[i*M*nnSample+j*nnSample+s] = k;
                            nnDist[i*M*nnSample+j*nnSample+s] = dist3D; // sqrt, not the squared one
                        }
                        s++;
                    }
                }
                radius += 0.05;
            }

            nnCount[i*M+j] = s<nnSample?s:nnSample;
        }
    }
}


// database:  B*N*3
// query:     B*M*3
// nnIndex:   B*M*nnOut
// nnDist:    B*M*nnOut
__global__ void cal_nnidx(int B, int N, int M, const float* database,
                          const float* query, int* nnIndex, float* nnDist)
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

            float best1=1e40, best2=1e40, best3=1e40, best4=1e40;
            int   besti1=0, besti2=0, besti3=0, besti4=0;

            for(int k=0;k<N;k++)
            {
                pt.x = database[i*N*3+k*3];
                pt.y = database[i*N*3+k*3+1];
                pt.z = database[i*N*3+k*3+2];

                delta.x = pt.x - ptQuery.x;
                delta.y = pt.y - ptQuery.y;
                delta.z = pt.z - ptQuery.z;

                float dist3D = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z; // squared 3D

                if (dist3D<best1) { // This is from PointNet++
                    best4=best3;
                    besti4=besti3;
                    best3=best2;
                    besti3=besti2;
                    best2=best1;
                    besti2=besti1;
                    best1=dist3D;
                    besti1=k;
                } else if (dist3D<best2) {
                    best4=best3;
                    besti4=besti3;
                    best3=best2;
                    besti3=besti2;
                    best2=dist3D;
                    besti2=k;
                } else if (dist3D<best3) {
                    best4=best3;
                    besti4=besti3;
                    best3=dist3D;
                    besti3=k;
                } else if (dist3D<best4) {
                    best4=dist3D;
                    besti4=k;
                }

                nnIndex[i*M*3+j*3]   = besti1;
                nnIndex[i*M*3+j*3+1] = besti2;
                nnIndex[i*M*3+j*3+2] = besti3;
                nnIndex[i*M*3+j*3+3] = besti4;
                nnDist[i*M*3+j*3]    = sqrtf(best1); // sqrt, not the squared one
                nnDist[i*M*3+j*3+1]  = sqrtf(best2); // sqrt, not the squared one
                nnDist[i*M*3+j*3+2]  = sqrtf(best3); // sqrt, not the squared one
                nnDist[i*M*3+j*3+3]  = sqrtf(best4); // sqrt, not the squared one
            }
        }
    }
}


void buildSphereNeighborLauncher(int B, int N, int M, int nnSample, float radius,
                                 const float* database, const float* query, int* nnIndex,
                                 int* nnCount, float* nnDist)
{
    cal_nnidx_sphere<<<B,1024>>>(B, N, M, nnSample, radius,
                                  database, query, nnIndex, nnCount, nnDist);
}

void buildNearestNeighborLauncher(int B, int N, int M, const float* database,
                                  const float* query, int* nnIndex, float* nnDist)
{
    cal_nnidx<<<B,1024>>>(B, N, M, database, query, nnIndex, nnDist);
}
