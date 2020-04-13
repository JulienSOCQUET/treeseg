//Andrew Burt - a.burt@ucl.ac.uk

#include "treeseg.h"

#include <pcl/io/pcd_io.h>

int main (int argc, char** argv)
{
	float zstep = atof(argv[1]);
	int nnearest = atoi(argv[2]);
 	char* cloud_fname = argv[3];


	pcl::PCDReader reader;
	pcl::PointCloud<PointTreeseg>::Ptr cloud(new pcl::PointCloud<PointTreeseg>);
	
	reader.read(cloud_fname,*cloud);
	std::vector<std::vector<float>> nndata;
	nndata = dNNz(cloud,nnearest,zstep);
	for(int i=0;i<nndata.size();i++) std::cout << nndata[i][0] << " " << nndata[i][1] << std::endl;
	return 0;
}
