package models.clustering;

import java.io.IOException;
import java.util.List;

import models.LDA.LDAHyperExtension;
import models.LSA.LatentSemanticAnalysis;
import models.LSA.SSpaceLSAShell;

import cc.mallet.types.InstanceList;
import cern.colt.matrix.DoubleMatrix1D;
import edu.ucla.sspace.vector.DoubleVector;

public class LDAKmeans {

	List<String> orgDocStrings = null;
	protected InstanceList instances = null;
	protected int[] clusterLabels;
	protected List<DoubleMatrix1D> centers = null;
	protected String initMethod;
	protected String distanceType = "Sphecial";
	protected int topicNumber = 20;
	protected int clusterNumber = 2;
	protected int seed = 0;
	
	public LDAKmeans(List<String> orgDocStrings, InstanceList instances, int topicNumber, int clusterNumber, int seed) {
		this.orgDocStrings = orgDocStrings;
		this.instances = instances;
		this.topicNumber = topicNumber;
		this.clusterNumber = clusterNumber;
		this.seed = seed;
	}
	
	public void estimate() throws IOException {
//		MalletParallelLDA lda = new MalletParallelLDA (topicNumber, 50.0, 0.01);
//		lda.printLogLikelihood = true;
////		lda.setTopicDisplay(50, 10);
//		lda.addInstances(instances);
//		lda.setNumThreads(1);
//		lda.estimate();
//		double[][] dataMatrix = lda.getDocumentTopicsData();
		
		LDAHyperExtension ldaHyper = new LDAHyperExtension (topicNumber, 50.0, 0.01);
        ldaHyper.isUseKmeansToInitialize(false);
        ldaHyper.addInstances(this.instances);
        ldaHyper.setNumIterations(1000);
        ldaHyper.setTopicDisplay(500, 100);
        ldaHyper.estimate();
        double[][] dataMatrix = ldaHyper.getDocumentTopicsData(true);
        
//        SSpaceLSAShell LSA = new SSpaceLSAShell();
//        LSA.run(this.orgDocStrings, topicNumber);
//        LatentSemanticAnalysis lsaSA = LSA.GetLSASenmanticAnalysis();
//        double[][] dataMatrix = new double[this.orgDocStrings.size()][];
//        for (int i = 0; i < this.orgDocStrings.size(); ++i) {
//        	DoubleVector vector = lsaSA.getDocumentVector(i);
//        	double[] vectorArray = new double[vector.length()];
//        	for (int j = 0; j < vector.length(); ++j) {
//        		vectorArray[j] = vector.get(j);
//        	}
//        	dataMatrix[i] = vectorArray;
//        }
       
		GeneralKmeans kmeans = new GeneralKmeans(dataMatrix, true, clusterNumber, "maxmin", seed);
		kmeans.setDistType("Euclidean");
		kmeans.setCenterSparse(false);
		kmeans.estimate();
		this.clusterLabels = kmeans.getLabels();
		this.centers = kmeans.getCentersList();
		
	}
	
	public int[] getLabels() {
		return clusterLabels;
	}
	
	public int getLabel(int index) {
		return clusterLabels[index];
	}

	public DoubleMatrix1D[] getCenters() {
		DoubleMatrix1D[] centersArray = new DoubleMatrix1D[centers.size()];
		for (int i = 0; i < centers.size(); ++i) {
			centersArray[i] = centers.get(i);
		}
		return centersArray;
	}

	public List<DoubleMatrix1D> getCentersList() {
		return centers;
	}
	
	public int getClusterNum() {
		return clusterNumber;
	}

	public void setClusterNum(int clusterNum) {
		this.clusterNumber = clusterNum;
	}
}
