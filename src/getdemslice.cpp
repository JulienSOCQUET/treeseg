//Andrew Burt - a.burt@ucl.ac.uk

#include "treeseg.h"

#include <pcl/io/pcd_io.h>
#include <omp.h>

int main (int argc, char *argv[])
{
	float resolution = atof(argv[1]);
	float zmin = atof(argv[2]);
	float zmax = atof(argv[3]);
	float grpercentile = atof(argv[4]);
	std::vector<std::string> id = getFileID(argv[5]);

	std::cout << "max num threads: " << omp_get_max_threads() << std::endl;

	pcl::PointCloud<PointTreeseg>::Ptr plotcloud(new pcl::PointCloud<PointTreeseg>);
	pcl::PCDWriter writer;
	std::cout << "Reading plotcloud..." << std::endl;
	readTiles(argc,argv,plotcloud);
	std::cout << "Finished reading plotcloud..." << std::endl;

	std::stringstream ss;
	ss.str("");
	ss << id[0] << ".slice.pcd";
	std::vector<std::vector<float>> dem;
	pcl::PointCloud<PointTreeseg>::Ptr slice(new pcl::PointCloud<PointTreeseg>);
	std::cout << "Running getDemAndSlice..." << std::endl;

	dem = getDemAndSlice(plotcloud,resolution,zmin,zmax,slice,grpercentile);
	// for(int j=0;j<dem.size();j++) std::cout << dem[j][0] << " " << dem[j][1] << " " << dem[j][2] << std::endl;
	std::cout << "Writing slice..." << std::endl;
	writer.write(ss.str(),*slice,true);
	return 0;
}
