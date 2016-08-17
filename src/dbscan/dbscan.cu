// This is a relatively unmodified example implementation of Bohm's GPU algorithm for performing
// dbscan. See "Density-based Clustering using Graphics Processors" Bohm 2009 for more
// details. 
// 
// Note: There are some limitations with this algorithm specifically in regards to maxseeds
//       and cluster collisions that may result in weird DBSCAN results with certain datasets.

/*****
 * CUDA-DCLUST: An implementation of DBSCAN in CUDA. 
 *
 * DBSCAN is a clustering algorithm that groups together data
 * that meets spatial locality and density charactoristics. 
 * The spatial locality and density critera used by DBSCAN is
 * set via the parameters "eps" and "minpts".
 *
 * The eps parameter is the spatial locality constraint for
 * DBSCAN. Eps defines the size of a point's neighborhood. 
 * It is used for determining what points are neighbors to
 * one another. A point is a neighbor to another point if
 * they are withing Eps distance of one another. 
 *  
 * The minpts parameters is the density constraint for DBSCAN.
 * Minpts defines the minimum number of neighbor points needed
 * for the formation of a cluster. If there are minpts number of
 * points (or greater) that fall within eps distance of a point,
 * a cluster is formed. 
 *
 * When a cluster is found by DBSCAN, all neighbor points to the origin
 * are expanded to fill in the cluster. The expansion operation
 * looks for the neighbor points to all neighbors of the origin point.
 * If any of those neighbor points have neighborhoods that exceed minpts,
 * those neighbor points are added to the cluster and subsequently expanded. 
 * If a point does not have at least minpts neighbors, expansion stops at that
 * point and none of its neighbors are added to the cluster. 
 *
 * Description of this implementation:
 *
 * The implementation here is a very slightly modified version of 
 * Bohm's (2009) implementation of DBSCAN in CUDA. The implementation
 * attempts to follow the psudocode example shown in Bohm's paper. 
 * There are known issues with the implementation in Bohm, so this
 * should be used only as a design reference and not for a real world
 * implementation. 
 *
 * Data for this example is generated randomly (using RAND). 
 * The number of data points is controlled by RANDOM_POINT_COUNT. 
 * Minpts and Eps are defined static values set in the variables 
 * minpts and eps respectively. 
 *
 * The GPU portion of this implementation operates by having every
 * CUDA thread block expand one point of the dataset. The expansion
 * operation in CUDA determines which points in the total dataset are 
 * neighbors to the one being expaned. This operation is performed
 * by using a simple euclidean distance calculation between points
 * in the dataset. If the distance is less than eps, the point is
 * considered a neighbor. If the number of neighbor points found is
 * greater than minpts, those points are marked as being members of
 * a cluster (marked in a states variable with the block id) and those
 * points are added to the seed list. If a collision is detected (the
 * point is already marked as a memeber of a cluster), the collision 
 * between the thread blocks is marked in the collision matrix indicating
 * the clusters found by these thread blocks should be merged by the CPU.
 * The iteration in the GPU continues until all seed lists are exhausted. 
 *
 * The CPU is responsible for marking the final clusters in the states 
 * variable, merging colliding clusters, and supplying points to the 
 * seed list of idle GPU thread blocks. 
 */ 
#include <stdio.h>      
#include <stdlib.h>     
#include <time.h>       
#include <vector>
#include <algorithm> 
#include <map>
#include <set>
// Number of random points to generate for testing.
#define RANDOM_POINT_COUNT 10000

// Maximum number of unprocessed points per cluster.
#define MAX_SEEDS 1024

// Number of CUDA thread blocks/thread count per block to use for processing
#define THREAD_BLOCKS 32
#define THREAD_COUNT 128

// Static values used for identifying unprocessed and noise (non-cluster member) points. 
#define UNPROCESSED -1
#define NOISE -2

/**
 * Standard GPU error checking function, application will exit on
 * any cuda error. 
 */
#define cuErrorCheck(ans) { errorAssert((ans), __FILE__, __LINE__); }
inline void errorAssert(cudaError_t retCode, const char *file, int line)
{
   if (retCode != cudaSuccess) 
   {
      fprintf(stderr,"GPU op Failed: %s %s %d\n", cudaGetErrorString(retCode), file, line);
      exit(retCode);
   }
}

/**********************************************
 *  Global GPU data structures and parameters *
 **********************************************/

// DBSCAN MinPts and Eps values
__device__ __constant__ int minpts = 5; 
__device__ __constant__ double eps = 0.1;

// Input Point Space, [n][D], n = pointID, d = position on dimension d.
// ex: point[4][0] = x dimension location of point 4. 
//     point[4][1] = y dimension location of point 4.
__device__ float points[RANDOM_POINT_COUNT][2];

// Track the point state (what cluster the point is in and whether or not it
// has been processed)
__device__ int pointState[RANDOM_POINT_COUNT];

// Seed list, a list of unprocessed points that are members of a cluster but
// have yet to be expanded. One needed
__device__ int seedList[THREAD_BLOCKS][MAX_SEEDS];
__device__ int curSeedLength[THREAD_BLOCKS];

// Collision Matrix, when two thread blocks collide (point has been processed
// as a cluster by someone else) mark the collisions and join the clusters.
__device__ int16_t collisionMatrix[THREAD_BLOCKS][THREAD_BLOCKS];

// Host variable pointer storage to CUDA globals. These are used to allow
// the CPU to perform memory transfers to/from the device. 
int16_t ** sym_collision;
int ** sym_seedList;
int * sym_curSeedLength;
float ** sym_points;
int * sym_pointState;

// Function definitions
bool SelectNextPointSet(std::vector<int> & pointsRemaining, int * clusterCount);
void FinalizeClusters(int * states, int * clusterCount);
__global__ void DBSCAN(void);
__device__ void markAsCandidate(int pid, int chainID);

int main() {
    int ret;

    // Generate random 2-D points for testing
    float point_gen[RANDOM_POINT_COUNT][2];
    std::vector<int> pointsRemaining;
    for (int x = 0; x < RANDOM_POINT_COUNT; x++){
        point_gen[x][0] =  static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/10));
        point_gen[x][1] =  static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/10));
        fprintf(stderr, "%f,%f\n", point_gen[x][0],point_gen[x][1]);
        pointsRemaining.push_back(x);
    }
    fprintf(stderr,"Generated %llu points\n", pointsRemaining.size());

    // Ensure that the GPU is functioning correctly. 
    cuErrorCheck( cudaFree(0) );

    // Get the CUDA device variable locations for the global variables defined
    // above. sym_ is how these variables will be manipulated in C. 
    cuErrorCheck(cudaGetSymbolAddress((void **)&sym_collision, collisionMatrix));
    cuErrorCheck(cudaGetSymbolAddress((void **)&sym_seedList, seedList));
    cuErrorCheck(cudaGetSymbolAddress((void **)&sym_curSeedLength, curSeedLength));
    cuErrorCheck(cudaGetSymbolAddress((void **)&sym_points, points));
    cuErrorCheck(cudaGetSymbolAddress((void **)&sym_pointState, pointState));

    // Initialize CUDA data structures.
    
    // Initialize the collision matrix. This matrix is needed to detect when two CUDA thread blocks
    // are expanding overlapping clusters so that they can be merged.
    // A value of -1 indicates no overlap, a value of 1 indicates that two thread blocks overlap.
    cudaMemset(sym_collision, -1, sizeof(int16_t) * THREAD_BLOCKS * THREAD_BLOCKS);

    // The seed list specifies the number of points left to be processed by a thread block. 
    // This list is needed because each execution of the DBSCAN kernel is only checking 
    // if a point is a member of a cluster, If it is a member all of its neighbors are 
    // added to seedList and are checked on subsequent executions. 
    cudaMemset(sym_seedList, -1, sizeof(int) * THREAD_BLOCKS * MAX_SEEDS);

    // Number of points to be processed in seedlist
    cudaMemset(sym_curSeedLength, 0, sizeof(int) * THREAD_BLOCKS);

    // Sets the pointState to be unprocessed for all points in the space. Unprocessed 
    // points are added to seedList by the CPU for processing by DBSCAN.
    cudaMemset(sym_pointState, UNPROCESSED, sizeof(int) * RANDOM_POINT_COUNT);

    // copy the randomly generated points to device readable memory
    cudaMemcpy(sym_points, point_gen, sizeof(float) * RANDOM_POINT_COUNT * 2, cudaMemcpyHostToDevice);

    // Initialize the found cluster count to be zero. 
    int clusterCount = 0;

    fprintf(stderr,"Starting DBSCAN\n");

    // Main running loop to perform DBSCAN. 
    // SelectNextPointSet initalizes the seedList and marks clusters identified by DBSCAN
    while (SelectNextPointSet(pointsRemaining, &clusterCount) == true) {
        cuErrorCheck(cudaDeviceSynchronize());

        // run the DBSCAN algorithm with the specified block and thread counts.
        DBSCAN<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT,1)>>>();

        cuErrorCheck(cudaDeviceSynchronize());

        // Print out the number of points remaining (that the CPU knows about)
        // to be checked by DBSCAN.
        fprintf(stderr, "Points Remaining: %llu\n", pointsRemaining.size());
    }   
}

/****
 * SelectNextPointSet does the following
 *
 * 1. Checks the seed list status for each thread block. If any 
 *    thread block has a seed list of length > 0, this function returns
 *    so that DBSCAN operation can be run again. 
 *
 * 2. If the seed list for all thread blocks is empty, look to see if any
 *    clusters were found. If so, finalize those clusters and reset the collision
 *    matrix. 
 * 
 * 3. Reset the seed list to new points to be processed. If points are left to
 *    be processed, return false indicating that the DBSCAN operation is complete. 
 */ 
bool SelectNextPointSet(std::vector<int> & pointsRemaining, int * clusterCount) {
    // Keeps track of the completion status of DBSCAN. 
    int complete = 0;
    bool refresh = true;

    // CPU local storage for seed counts. 
    int lseedCount[THREAD_BLOCKS];
    memset(lseedCount, 0, sizeof(int) * THREAD_BLOCKS);

    // Copy the seed counts for the threadblocks from the GPU to the CPU. 
    cudaMemcpy(lseedCount, sym_curSeedLength, sizeof(int) * THREAD_BLOCKS, cudaMemcpyDeviceToHost);

    // Check the seed lengths, if any are > 0, return to continue running DBSCAN
    for (int i = 0; i < THREAD_BLOCKS; i++) {
        if (lseedCount[i] > 0) {
            refresh = false;
            break;
        } 
    }

    if (refresh == false)
        return true;

    // Local storage fo the point states.        
    int lpointStates[RANDOM_POINT_COUNT];

    // Copy the current point states from GPU memory to local memory
    cudaMemcpy(lpointStates, sym_pointState, sizeof(int) * RANDOM_POINT_COUNT, cudaMemcpyDeviceToHost);

    // Mark clusters that have been found in the last iteration of DBSCAN.
    FinalizeClusters(lpointStates, clusterCount);
    
    // Local storage for the thread block seed lists
    int pointSeeds[THREAD_BLOCKS][MAX_SEEDS];

    // Copy the current seed list from the GPU to the CPU
    cudaMemcpy(pointSeeds, sym_seedList, sizeof(int) * THREAD_BLOCKS * MAX_SEEDS, cudaMemcpyDeviceToHost);

    // Select new points to process and place them into the seed list for 
    // each thread block. 
    for (int i = 0; i < THREAD_BLOCKS; i++) {
        bool found = false;
        while(!pointsRemaining.empty()) {
            int pos = pointsRemaining.back();
            pointsRemaining.pop_back();
            //fprintf(stderr,"%d,%d\n", lpointStates[pos], UNPROCESSED);
            if ( lpointStates[pos] == UNPROCESSED) {
                lseedCount[i] = 1;
                pointSeeds[i][0] = pos;
                found = true;
                break;
            }
        }
        if (found == false) {
            complete++;
        }
    }

    // State how many clusters have been found and how many points are remaining. 
    fprintf(stderr, "Found %d Clusters, %llu points remaining\n", *clusterCount, pointsRemaining.size());

    // Copy seed length, point states, and seed list back to the GPU
    cudaMemcpy(sym_curSeedLength, lseedCount, sizeof(int) * THREAD_BLOCKS, cudaMemcpyHostToDevice);
    cudaMemcpy(sym_pointState, lpointStates, sizeof(int) * RANDOM_POINT_COUNT, cudaMemcpyHostToDevice);
    cudaMemcpy(sym_seedList, pointSeeds, sizeof(int) * THREAD_BLOCKS * MAX_SEEDS, cudaMemcpyHostToDevice);

    // If we have processed all points, return false. 
    if (complete == THREAD_BLOCKS) 
        return false;

    return true;

}

/**
 * FinalizeClusters identifies the clusters found by DBSCAN, 
 * handles collisions between thread blocks, and finalizes the
 * clusters. 
 */
void FinalizeClusters(int * states, int * clusterCount) {
    // Local storage for the collision matrix. 
    int16_t localCol[THREAD_BLOCKS][THREAD_BLOCKS];

    // Copy the collision matrix from the GPU to the CPU. 
    cudaMemcpy(localCol, sym_collision, sizeof(int16_t) * THREAD_BLOCKS * THREAD_BLOCKS, cudaMemcpyDeviceToHost);
    
    // Collision map is a map identifying what thread blocks are expanding
    // overlapping clusters. 
    std::map<int,int> colMap;
    std::set<int> blockSet;
    for (int i = 0; i < THREAD_BLOCKS; i++) {
        colMap[i] = i;
        blockSet.insert(i);
    }

    // Iterate over the collision matrix, marking colliding
    // thread blocks in colMap. This code needs to be explained more
    // and can likely be made more efficient...
    std::set<int>::iterator it;
    do {
        it = blockSet.begin();
        int curBlock = *it; 
        std::set<int> expansionQueue; 
        std::set<int> finalQueue;
        finalQueue.insert(curBlock);
        expansionQueue.insert(curBlock);
        do {
            it = expansionQueue.begin();
            int expandBlock = *it;
            expansionQueue.erase(it);
            blockSet.erase(expandBlock);
            for (int x = 0; x < THREAD_BLOCKS; x++){
                if (x == expandBlock)
                    continue;
                if ((localCol[expandBlock][x] == 1 || localCol[x][expandBlock]) 
                    && blockSet.find(x) != blockSet.end()) {
                    expansionQueue.insert(x);
                    finalQueue.insert(x);
                }
            }
        } while (expansionQueue.empty() == 0);

        for (it = finalQueue.begin(); it != finalQueue.end(); ++it) {
            colMap[*it] = curBlock;
        }
    } while(blockSet.empty() == 0);

    // Mark all points that were found in this iteration of DBSCAN as members of
    // a cluster. 
    std::vector<std::vector<int> > clusters(THREAD_BLOCKS, std::vector<int>());
    for (int i = 0; i < RANDOM_POINT_COUNT; i++) {
        if (states[i] >= 0 && states[i] < THREAD_BLOCKS) {
            clusters[colMap[states[i]]].push_back(i);
        }
    }

    // Count the clusters and update the point state variable.
    for (int i = 0; i < clusters.size(); i++) {
        if (clusters[i].size() == 0)
            continue;
        for(int x = 0; x < clusters[i].size(); x++) {
            states[clusters[i][x]] = *clusterCount + THREAD_BLOCKS + 1;
        }
        (*clusterCount)++;
    }

    // Copy a blank collision matrix back to the GPU. 
    cudaMemset(sym_collision, -1, sizeof(int16_t) * THREAD_BLOCKS * THREAD_BLOCKS);
    printf("Cluster Count:%d\n", *clusterCount);
}


// This is the DBSCAN kernel that runs in the GPU. 
__global__ void DBSCAN(void){
    // shared variable between all threads in the block.
    __shared__ int leave;

    // Stores points in the neighborhood, up to max seeds. 
    __shared__ int nhood[MAX_SEEDS];

    // Stores the number of points currently in the neighborhood.
    __shared__ int nhoodCount;

    // The X,Y values for the point that this thread block is processing
    __shared__ float point[2];

    // The point id (possition in the points array) for the point to 
    // be processed. 
    __shared__ int pointID;

    // Get the current block id, we use the blocks position on the x access
    // of the block grid for its id. This makes the assumption that a 1 dimension
    // block grid is in use. 
    int chainID = blockIdx.x;

    // Get the current seed length
    int seedLength = curSeedLength[chainID];

    // Set neighborhood count to zero
    nhoodCount = 0;

    // If we have nothing to process (no seeds), exit.
    if (seedLength == 0)
        return;

    // otherwise decrement seed length to get the current point.
    seedLength = seedLength - 1;
    
    // Get the point id of the point to process from seedlength.
    // This value will be identical across all threads in the block. 
    pointID = seedList[chainID][seedLength];

    // Get the X/Y values for the point. 
    point[0] = points[pointID][0];
    point[1] = points[pointID][1];

    leave = 0;

    // Synchronize all threads in this block. 
    __syncthreads();

    // If this point has already been processed. Set leave and exit. 
    if (threadIdx.x == 0) {
        curSeedLength[chainID] = seedLength;
        if (pointState[pointID] != UNPROCESSED)
            leave = 1;
    }
    __syncthreads();
    if (leave == 1)
        return;
    
    __syncthreads();

    // Check all points in the dataset looking for points in this points area.
    // Each thread is comparing a different point in the dataset to the point
    // this thread block is processing. 
    for(int i = threadIdx.x; i < RANDOM_POINT_COUNT; i = i + THREAD_COUNT) {
        register float comp[2];
        comp[0] = points[i][0];
        comp[1] = points[i][1];

        // compute the distance between the point this threadblock is processing
        // to another point in the dataset. 
        register float delta = 0;
        delta = sqrtf(powf(comp[0]  - point[0], 2) + powf(comp[1] - point[1],2));

        // If this point is closer than eps distance away.
        if (delta <= eps) {
            // Add a point the the neighborhood.
            register int h = atomicAdd(&nhoodCount,1);

            // If the neighborhood count is already greater than minpts, 
            // mark the point as a member of the cluster immediately.
            if (nhoodCount >= minpts) {
                markAsCandidate(i, chainID);
            }
            // Otherwise hold the point for later processing 
            else {
                nhood[h] = i;
            }
        }
    }

    __syncthreads();
    // Reset neighborhood count to minpts if this value is greater.
    // We should only have < minpts in the array nhood. 
    if (threadIdx.x == 0 && nhoodCount > minpts) {
        nhoodCount = minpts;
    }
    __syncthreads();

    // Return the to points in nhood if there are more than minpts neighbors
    // and mark them as members of the cluster. 
    if (nhoodCount >= minpts) {
        pointState[pointID] = chainID;
        for (int i = threadIdx.x; i < nhoodCount; i = i + THREAD_COUNT) {
            markAsCandidate(nhood[i], chainID);
        }
    // Otherwise mark the point we are processing as noise. 
    } else  {
        pointState[pointID] = NOISE;
    }
    __syncthreads();
    // Finally make sure that seed length is not overflowing. 
    if (threadIdx.x == 0 && curSeedLength[chainID] >= MAX_SEEDS) {
        curSeedLength[chainID] = MAX_SEEDS - 1;
    }
}

// Mark a point as a member of a cluster. 
__device__ void markAsCandidate(int pid, int chainID) {
    // Set the state of the point to be a member of a cluster. 
    // oldState returns the previous state value.
    register int oldState = atomicCAS(&(pointState[pid]), 
                                      UNPROCESSED, chainID);    
    // If the old state is UNPROCESSED, add to the queue to be 
    // processed by subsequent DBSCAN iterations.
    if (oldState == UNPROCESSED) {
        register int h = atomicAdd(&(curSeedLength[chainID]), 1);
        if (h < MAX_SEEDS) {
            seedList[chainID][h] = pid;
        } 
    // If we have a collision, mark the collision matrix.
    } else if (oldState != NOISE && oldState != chainID && oldState < THREAD_BLOCKS) {
        if (oldState < chainID) {
            collisionMatrix[oldState][chainID] = 1;
        } else {
            collisionMatrix[chainID][oldState] = 1;
        }
    // Handle an unhandled edge case in Bohm's code for NOISE points that become members
    // of a cluster via expansion of another point. 
    } else if (oldState == NOISE) {
        oldState = atomicCAS(&(pointState[pid]), NOISE, chainID);           
    }
}
