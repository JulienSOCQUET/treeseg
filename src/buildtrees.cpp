//Andrew Burt - a.burt@ucl.ac.uk

#include <pcl/io/pcd_io.h>

#include <armadillo>

#include <chrono>

#include "treeseg.h"

#include <sys/stat.h>
struct VolumeFile
{
	char *name;
	size_t size;
	bool operator<(const VolumeFile &str) const
	{
		return (size < str.size);
	}
};

long GetFileSize(std::string filename)
{
	struct stat stat_buf;
	int rc = stat(filename.c_str(), &stat_buf);
	return rc == 0 ? stat_buf.st_size : -1;
}

int main(int argc, char *argv[])
{

	std::cout << "max num threads: " << omp_get_max_threads() << std::endl;

	float smoothness = atof(argv[1]);
	bool sepwoodleaf = atoi(argv[2]);
	int start_argc = 3;

	pcl::PCDReader reader;
	pcl::PCDWriter writer;
	std::stringstream ss;
	std::vector<pcl::PointCloud<PointTreeseg>::Ptr> trees;
	// #pragma omp parallel shared(trees)
	// 	{

	auto start_time = std::chrono::steady_clock::now();

	std::vector<VolumeFile> volumes;

	for (int i = start_argc; i < argc; i++)
	{
		long int filesize = 0;
		filesize = GetFileSize(argv[i]);
		VolumeFile volume;
		volume.name = argv[i];
		volume.size = filesize;
		volumes.push_back(volume);
		std::cout << "Size: " << volume.size << std::endl;
	}

	std::sort(volumes.begin(), volumes.end());

// #pragma omp parallel for schedule(dynamic, 1)
#pragma omp parallel
	{
#pragma omp single
		{
			for (int i = 0; i < volumes.size(); i++)
			{

				std::cout << "Reading: " << volumes[i].name << std::endl;

				pcl::PointCloud<PointTreeseg>::Ptr cluster(new pcl::PointCloud<PointTreeseg>);
				reader.read(volumes[i].name, *cluster);
				std::vector<std::string> id = getFileID(volumes[i].name);

				std::cout << "Region-based segmentation: " << std::endl;
				std::vector<pcl::PointCloud<PointTreeseg>::Ptr> regions;
				int nnearest = 50;
				int nmin = 3;
				regionSegmentation(cluster, nnearest, nmin, smoothness, regions);
				ss.str("");
				ss << id[0] << ".ec.rg." << id[1] << ".pcd";
				writeClouds(regions, ss.str(), false);
				std::cout << ss.str() << std::endl;
				//
				// int ngausians = 5;
				// if ((sepwoodleaf == true) && (regions.size() >= ngausians))
				// {
				// 	std::cout << "Leaf stripping: " << std::endl;
				// 	//
				// 	std::cout << " Region-wise, " << std::flush;
				// 	arma::mat rfmat;
				// 	arma::gmm_full rmodel;
				// 	gmmByCluster(regions, ngausians, 1, 5, 50, 100, rfmat, rmodel);
				// 	std::cout << "Done Gmm by Cluster " << std::endl;

				// 	std::vector<int> rclassifications;
				// 	rclassifications = classifyGmmClusterModel(regions, 5, rfmat, rmodel);
				// 	std::vector<pcl::PointCloud<PointTreeseg>::Ptr> csepclouds;
				// 	separateCloudsClassifiedByCluster(regions, rclassifications, csepclouds);
				// 	ss.str("");
				// 	ss << id[0] << ".ec.rg.rlw." << id[1] << ".pcd";
				// 	writeCloudClassifiedByCluster(regions, rclassifications, ss.str());
				// 	std::cout << ss.str() << std::endl;
				// 	//
				// 	std::cout << " Point-wise, " << std::flush;
				// 	arma::mat pfmat;
				// 	arma::gmm_diag pmodel;
				// 	gmmByPoint(csepclouds[1], 50, 5, 1, 5, 50, 100, pfmat, pmodel);
				// 	std::vector<int> pclassifications;
				// 	pclassifications = classifyGmmPointModel(csepclouds[1], 5, pfmat, pmodel);
				// 	std::vector<pcl::PointCloud<PointTreeseg>::Ptr> psepclouds;
				// 	separateCloudsClassifiedByPoint(csepclouds[1], pclassifications, psepclouds);
				// 	ss.str("");
				// 	ss << id[0] << ".ec.rg.rlw.plw." << id[1] << ".pcd";
				// 	writeCloudClassifiedByPoint(csepclouds[1], pclassifications, ss.str());
				// 	std::cout << ss.str() << std::endl;
				// 	//
				// 	ss.str("");
				// 	ss << id[0] << ".ec.rg.rlw.plw.w." << id[1] << ".pcd";
				// 	pcl::PointCloud<PointTreeseg>::Ptr wood(new pcl::PointCloud<PointTreeseg>);
				// 	*wood += *csepclouds[0] + *psepclouds[0];
				// 	writer.write(ss.str(), *wood, true);
				// 	std::cout << ss.str() << std::endl;
				// 	//
				// 	std::cout << "Re-segmenting regions: " << std::flush;
				// 	regions.clear();
				// 	regionSegmentation(wood, nnearest, nmin, smoothness + 2.5, regions);
				// 	ss.str("");
				// 	ss << id[0] << ".ec.rg.rlw.plw.w.rg." << id[1] << ".pcd";
				// 	writeClouds(regions, ss.str(), false);
				// 	std::cout << ss.str() << std::endl;
				// }
				//
				std::cout << "Optimising regions: " << std::flush;
				removeFarRegions(regions);
				ss.str("");
				if (sepwoodleaf == true)
					ss << id[0] << ".ec.rg.rlw.plw.w.rg.o." << id[1] << ".pcd";
				else
					ss << id[0] << ".ec.rg.o." << id[1] << ".pcd";
				writeClouds(regions, ss.str(), false);
				std::cout << ss.str() << std::endl;
				//
				// start_time = std::chrono::steady_clock::now();

				std::cout << "Building tree: " << std::flush;
				pcl::PointCloud<PointTreeseg>::Ptr tree(new pcl::PointCloud<PointTreeseg>);

				buildTree(regions, tree, id[1]);

				ss.str("");
				ss << id[0] << "_" << id[1] << ".pcd";
				writer.write(ss.str(), *tree, true);
				std::cout << ss.str() << std::endl;

#pragma omp critical(addtrees)
				{
					trees.push_back(tree);

					ss.str("");
					ss << id[0] << ".all_trees.pcd";
					writeClouds(trees, ss.str(), false);
				}
			}
		}
	}

	auto end_time = std::chrono::steady_clock::now();

	std::cout << "Elapsed time in seconds : "
			  << std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count()
			  << " sec" << std::endl;

	return 0;
}
