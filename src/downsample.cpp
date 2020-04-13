//Andrew Burt - a.burt@ucl.ac.uk

#include "treeseg.h"

#include <pcl/io/pcd_io.h>

int main(int argc, char *argv[])
{
	std::cout << "max num threads: " << omp_get_max_threads() << std::endl;

	float edgelength = atof(argv[1]);
	int start_argc = 2;

	pcl::PCDReader reader;
	pcl::PCDWriter writer;

#pragma omp parallel for
	for (int i = 2; i < argc; i++)
	{
		std::stringstream ss;
		pcl::PointCloud<PointTreeseg>::Ptr original(new pcl::PointCloud<PointTreeseg>);
		reader.read(argv[i], *original);
		std::vector<std::string> id = getFileID(argv[i]);
		pcl::PointCloud<PointTreeseg>::Ptr filtered(new pcl::PointCloud<PointTreeseg>);
		downsample_byOctTree(original, edgelength, filtered);
		ss.str("");
		ss << id[0] << ".tile.downsample." << id[1] << ".pcd";
		writer.write(ss.str(), *filtered, true);

#pragma omp critical
		{
			std::cout << "Done With: " << argv[i] << "\t size:  " << original->points.size() << std::endl;
		}
	}
	return 0;
}
