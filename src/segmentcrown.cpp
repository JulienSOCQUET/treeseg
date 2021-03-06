//Andrew Burt - a.burt@ucl.ac.uk

#include "leafsep.h"

#include <pcl/io/pcd_io.h>

#include <armadillo>

#include <chrono>

int main(int argc, char *argv[])
{

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

#pragma omp parallel for schedule(dynamic, 1)
	for (int i = start_argc; i < argc; i++)
	{

		std::cout << "----------: " << argv[i] << std::endl;
		std::cout << "Reading volume cloud: " << std::flush;
		std::vector<std::string> id = getFileID(argv[i]);
		pcl::PointCloud<PointTreeseg>::Ptr volume(new pcl::PointCloud<PointTreeseg>);
		reader.read(argv[i], *volume);
		std::cout << "complete" << std::endl;
		//
		std::cout << "Euclidean clustering: " << std::flush;
		std::vector<std::vector<float>> nndata = dNNz(volume, 50, 2);
		float nnmax = 0;
		for (int k = 0; k < nndata.size(); k++)
			if (nndata[k][1] > nnmax)
				nnmax = nndata[k][1];
		std::cout << nnmax << ", " << std::flush;
		std::vector<pcl::PointCloud<PointTreeseg>::Ptr> clusters;
		euclideanClustering(volume, nnmax, 3, clusters);
		ss.str("");
		ss << id[0] << ".ec." << id[1] << ".pcd";
		writeClouds(clusters, ss.str(), false);
		std::cout << ss.str() << std::endl;
		//
		std::vector<pcl::PointCloud<PointTreeseg>::Ptr> regions;
		int idx = findPrincipalCloudIdx(clusters);
		int nnearest = 50;
		int nmin = 3;

		ss.str("");
		ss << id[0] << ".ec.prim." << id[1] << ".pcd";
		writer.write(ss.str(),*clusters[idx], true);

		std::cout << ss.str() << std::endl;
		
		int ngausians = 5;
		if ((sepwoodleaf == true) && (regions.size() >= ngausians))
		{
			std::cout << "Region-based segmentation: " << std::flush;
			regionSegmentation(clusters[idx], nnearest, nmin, smoothness, regions);
			ss.str("");
			ss << id[0] << ".ec.rg." << id[1] << ".pcd";
			writeClouds(regions, ss.str(), false);
			std::cout << ss.str() << std::endl;
			//

			std::cout << "Leaf stripping: " << std::endl;
			//
			std::cout << " Region-wise, " << std::flush;
			arma::mat rfmat;
			arma::gmm_full rmodel;
			gmmByCluster(regions, ngausians, 1, 5, 50, 100, rfmat, rmodel);
			std::cout << "Done Gmm by Cluster " << std::endl;

			std::vector<int> rclassifications;
			rclassifications = classifyGmmClusterModel(regions, 5, rfmat, rmodel);
			std::vector<pcl::PointCloud<PointTreeseg>::Ptr> csepclouds;
			separateCloudsClassifiedByCluster(regions, rclassifications, csepclouds);
			ss.str("");
			ss << id[0] << ".ec.rg.rlw." << id[1] << ".pcd";
			writeCloudClassifiedByCluster(regions, rclassifications, ss.str());
			std::cout << ss.str() << std::endl;
			//
			std::cout << " Point-wise, " << std::flush;
			arma::mat pfmat;
			arma::gmm_diag pmodel;
			gmmByPoint(csepclouds[1], 50, 5, 1, 5, 50, 100, pfmat, pmodel);
			std::vector<int> pclassifications;
			pclassifications = classifyGmmPointModel(csepclouds[1], 5, pfmat, pmodel);
			std::vector<pcl::PointCloud<PointTreeseg>::Ptr> psepclouds;
			separateCloudsClassifiedByPoint(csepclouds[1], pclassifications, psepclouds);
			ss.str("");
			ss << id[0] << ".ec.rg.rlw.plw." << id[1] << ".pcd";
			writeCloudClassifiedByPoint(csepclouds[1], pclassifications, ss.str());
			std::cout << ss.str() << std::endl;
			//
			ss.str("");
			ss << id[0] << ".ec.rg.rlw.plw.w." << id[1] << ".pcd";
			pcl::PointCloud<PointTreeseg>::Ptr wood(new pcl::PointCloud<PointTreeseg>);
			*wood += *csepclouds[0] + *psepclouds[0];
			writer.write(ss.str(), *wood, true);
			std::cout << ss.str() << std::endl;
		}
	}

	auto end_time = std::chrono::steady_clock::now();

	std::cout << "Elapsed time in seconds : "
			  << std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count()
			  << " sec" << std::endl;
	return 0;
}
