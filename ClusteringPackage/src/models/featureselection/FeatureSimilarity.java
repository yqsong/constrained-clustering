package models.featureselection;

import java.util.ArrayList;
import java.util.logging.Logger;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;

/**
 * Unsupervised feature selection using mutual information and maximal spanning tree
 * 
 * See
 * @article{Mitra02,
    author = {Mitra, P.  and Murthy, C. A.  and Pal, S. K. },
    journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
    number = {3},
    pages = {301--312},
    title = {Unsupervised feature selection using feature similarity},
    volume = {24},
    year = {2002}
	};
 * 
 * @author Yangqiu Song @ IBM CRL
 */

public class FeatureSimilarity {
	
	private DoubleMatrix2D fMat = null;
	private DoubleMatrix2D fDistances = null;
	private static Logger logger = Logger.getLogger("com.ibm.feature.selection");	

	public FeatureSimilarity (DoubleMatrix2D features) {
		fMat = features;
		fDistances = new DenseDoubleMatrix2D(fMat.columns(), fMat.columns());
	}
	
	private double computeEuclidean(DoubleMatrix1D v1, DoubleMatrix1D v2) {
		assert(v1.size() == v2.size());
		
		double norm1 = v1.zDotProduct(v1);
		double norm2 = v2.zDotProduct(v2);
		double dot = v1.zDotProduct(v2);
				
		return Math.sqrt(norm1 + norm2 - 2 * dot);
	}
	
	private double computeSpecialEuclidean(DoubleMatrix1D v1, DoubleMatrix1D v2) {
		assert(v1.size() == v2.size());
		
		double norm1 = v1.zDotProduct(v1);
		double norm2 = v2.zDotProduct(v2);
		double dot = v1.zDotProduct(v2);
				
		return (1 - dot / ( Math.sqrt(norm1 * norm2) + Double.MIN_VALUE ) );
	}
	
	private double computeCorrelation(DoubleMatrix1D v1, DoubleMatrix1D v2) {
		assert(v1.size() == v2.size());
		int size = v1.size();

		double dot = v1.zDotProduct(v2);
		double v1mean = 0;
		double v2mean = 0;
		double v1square = 0;
		double v2square = 0;
		for (int i = 0; i < size; ++i) {
			v1mean += v1.getQuick(i);
			v2mean += v2.getQuick(i);
			v1square += v1.getQuick(i) * v1.getQuick(i);
			v2square += v2.getQuick(i) * v2.getQuick(i);
		}
		
		return  1 - Math.abs( (size * dot - v1mean * v2mean) /
				  ( Math.sqrt(size * v1square - v1mean * v1mean) + Double.MIN_VALUE) /
				  ( Math.sqrt(size * v2square - v2mean * v2mean) + Double.MIN_VALUE) );
	}

	
	private double computeSquareLossofPCA(DoubleMatrix1D v1, DoubleMatrix1D v2) {
		assert(v1.size() == v2.size());

		int size = v1.size();

		double dot = v1.zDotProduct(v2);
		double v1mean = 0;
		double v2mean = 0;
		double v1square = 0;
		double v2square = 0;
		for (int i = 0; i < size; ++i) {
			v1mean += v1.getQuick(i);
			v2mean += v2.getQuick(i);
			v1square += v1.getQuick(i) * v1.getQuick(i);
			v2square += v2.getQuick(i) * v2.getQuick(i);
		}
		
		double rho =  ( (size * dot - v1mean * v2mean) /
					  	(Math.sqrt(size * v1square - v1mean * v1mean) + Double.MIN_VALUE) /
					  	(Math.sqrt(size * v2square - v2mean * v2mean) + Double.MIN_VALUE) );
		
		v1mean /= size;
		v2mean /= size;
		
		double v1var = 0;
		double v2var = 0;
		for (int i = 0; i < size; ++i) {
			v1var += (v1.getQuick(i) - v1mean) * (v1.getQuick(i) - v1mean);
			v2var += (v2.getQuick(i) - v2mean) * (v2.getQuick(i) - v2mean);
		}
		v1var /= size;
		v2var /= size;
		
		return v1var + v2var - 
				Math.sqrt((v1var + v2var) * (v1var + v2var) - 4 * v1var * v2var * (1 - rho * rho));
		
	}
	
	private void computePairwiseSimilarity(String type) {
		if (this.fMat == null) {
			logger.warning("No feature loaded");
			return;
		}
		for (int i = 0; i < fMat.columns(); ++i) {
			for (int j = i + 1; j < fMat.columns(); ++j) {
				DoubleMatrix1D v1 = fMat.viewColumn(i);
				DoubleMatrix1D v2 = fMat.viewColumn(j);
				double distances = 0;
				if (type.equalsIgnoreCase("Euclidean")) {
					distances = computeEuclidean(v1, v2);
				} else if (type.equalsIgnoreCase("Special")) {
					distances = computeSpecialEuclidean(v1, v2);					
				} else if (type.equalsIgnoreCase("Correlation")) {
					distances = computeCorrelation(v1, v2);									
				} else if (type.equalsIgnoreCase("MaxInfoCompress")) {
					distances = computeSquareLossofPCA(v1, v2);					
				} else {
					logger.warning("No found simiarity type");
				}
				fDistances.setQuick(i, j, distances);
				fDistances.setQuick(j, i, distances);
//				System.out.println("row: " + i + " column: " + j + " value: " +  distances);
			}// end for j
		}// end for i
		
	}
	
	private class FeatureEdge implements Comparable<FeatureEdge> {
		private int srcIndex;
		private int dstIndex;
		private double distance;
		FeatureEdge (int from, int to, double dist) {
			srcIndex = from;
			dstIndex = to;
			distance = dist;
		}
		
		public void setSrc (int from) {
			srcIndex = from;
		}
		public void setDst (int to) {
			dstIndex = to;
		}
		public void setDistance (int dist) {
			distance = dist;
		}
		
		public int getSrc () {
			return srcIndex;
		}
		public int getDst () {
			return dstIndex;
		}
		public double getDistance () {
			return distance;
		}		
		
		public int compareTo(FeatureEdge arg0) {
			if (distance < arg0.distance)
			    return -1;
			else if (distance == arg0.distance)
			    return 0;
			else return 1;
		}

		public boolean equals(FeatureEdge arg0) {
			if (distance == arg0.distance
					&& srcIndex == arg0.srcIndex
					&& dstIndex == arg0.dstIndex)
			    return true;
			else 
				return false;
		}

	}
	
	/*
	 * See
	 * @article{Mitra02,
	    author = {Mitra, P.  and Murthy, C. A.  and Pal, S. K. },
	    journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
	    number = {3},
	    pages = {301--312},
	    title = {Unsupervised feature selection using feature similarity},
	    volume = {24},
	    year = {2002}
		};
	 * With a little modification!!
	 * A most easy way to remove the most similar feature in each step
	 * @author Yangqiu Song @ IBM CRL
	 */
	public int[] rankFeatures(String type) {
		if (this.fMat == null) {
			logger.warning("No feature loaded");
			return null;
		}
		
//		System.out.println("row: " + fMat.rows());
//		System.out.println("column: " + fMat.columns());
	
		computePairwiseSimilarity(type);
		
		int[] sortedIndices = new int[fMat.columns()];
		
		ArrayList<Integer> remainIndexList = new ArrayList<Integer>();
		ArrayList<ArrayList<FeatureEdge>> minIndexList = new ArrayList<ArrayList<FeatureEdge>>();
		for (int i = 0; i < fMat.columns(); ++i) {
			remainIndexList.add(i);
			ArrayList<FeatureEdge> minIndex = new ArrayList<FeatureEdge>();
			minIndexList.add(minIndex);
		}
		for (int i = 0; i < fDistances.rows(); ++i) {
			for (int j = 0; j < fDistances.columns(); ++j) {
				if (j != i) {
					FeatureEdge fEdge = new FeatureEdge(i, j, fDistances.get(i, j));
					// find place to insert
					int k; 
					for(k = 0; k < minIndexList.get(i).size(); ++k)          
						if( fEdge.compareTo(minIndexList.get(i).get(k)) < 0 )
							break;
					minIndexList.get(i).add(k, fEdge);
				}
			}
		}		
		
//		for (int i = 0; i < minIndexList.size(); ++i) {
//			for (int j = 0; j < minIndexList.get(i).size(); ++j) {
//				double value = minIndexList.get(i).get(j).getDistance();
//				int src = minIndexList.get(i).get(j).getSrc();
//				int dst = minIndexList.get(i).get(j).getDst();
//				System.out.println("src: " + src + " dst: " + dst + " dist: " + value);
//			}
//		}		
		int k = minIndexList.size() - 1;
		while (minIndexList.size() > 1) {
			double epsilon = Double.MAX_VALUE;
			int srcIndex = 0;
			int dstIndex = 0;
			for (int i = 0; i < minIndexList.size(); ++i) {
				if (minIndexList.get(i).get(0).getDistance() < epsilon) {
					epsilon = minIndexList.get(i).get(0).getDistance();
					srcIndex = minIndexList.get(i).get(0).getSrc();
					dstIndex = minIndexList.get(i).get(0).getDst();					
				}
			}
			
			int index = remainIndexList.indexOf(dstIndex);
			remainIndexList.remove(index);
			minIndexList.remove(index);
			System.out.println("row remove index " + index + " src " + srcIndex + " dst " + dstIndex + " " 
					+ " distance " + epsilon);
			for (int i = 0; i < minIndexList.size(); ++i) {
				for (int j = 0; j < minIndexList.get(i).size(); ++j) {
					int tempDstIndex = minIndexList.get(i).get(j).getDst();
					if (tempDstIndex == dstIndex) {
						int tempSrcIndex = minIndexList.get(i).get(j).getSrc();
						double value = fDistances.get(tempSrcIndex, dstIndex);
						minIndexList.get(i).remove(j);
//						FeatureEdge fEdge = new FeatureEdge(tempSrcIndex, dstIndex, fDistances.get(tempSrcIndex, dstIndex));
//						int temp = minIndexList.get(i).indexOf(fEdge);
//						boolean isOK = minIndexList.get(i).remove(fEdge);
						System.out.println("column remove index " + j + " src " + tempSrcIndex + " dst " + dstIndex + " " 
								+ " distance " + value);
		//				if (isOK != true) {
		//					System.out.println("Error");
		//				}
					}
				}
			}
			sortedIndices[k] = dstIndex;
			k--;
		
		}
		sortedIndices[0] = remainIndexList.indexOf(0);
		
		return sortedIndices;
	}
	
	// mini test
	public static void main (String[] args) {
		/*		double[][] feature = {{1, 0, 0},
							  {1, 0, 0},
							  {1, 0, 0},
							  {1, 0, 0},
							  {1, 0, 0},
							  {0, 1, 0},
							  {0, 1, 0},
							  {0, 1, 0},
							  {0, 1, 0},
							  {0, 1, 0},
							  {0, 0, 1},
							  {0, 0, 1},};	*/
/*		double[][] feature = {
				  {1, 1, 0, 1, 1, 0},
				  {1, 1, 0, 1, 1, 0},
				  {1, 1, 0, 1, 0, 1},
				  {1, 1, 0, 1, 1, 1},
				  {0, 0, 1, 1, 1, 1},
				  {0, 0, 1, 0, 0, 1},
				  {0, 0, 1, 0, 1, 1},
				  {0, 0, 1, 0, 1, 1}};	*/
		double[][] feature = {
				  {1, 1, 0, 1, 1, 0, 1},
				  {1, 1, 0, 1, 1, 0, 1},
				  {1, 1, 0, 1, 0, 1, 1},
				  {1, 1, 0, 1, 1, 1, 1},
				  {0, 0, 1, 1, 1, 1, 1},
				  {0, 0, 1, 0, 0, 1, 1},
				  {0, 0, 1, 0, 1, 1, 1},
				  {0, 0, 1, 0, 1, 1, 1}};	
/*		double[][] feature = {
				  {1, 3, 0, 2, 1, 0, 1},
				  {7, 1, 0, 1, 3, 0, 1},
				  {7, 4, 0, 2, 0, 1, 1},
				  {3, 1, 0, 1, 1, 1, 1},
				  {0, 0, 4, 1, 1, 2, 1},
				  {0, 0, 4, 0, 0, 2, 1},
				  {0, 0, 1, 0, 3, 1, 1},
				  {0, 0, 1, 0, 1, 1, 1}};	*/
/*		double[][] feature = {
				  {11, 0, 1},
				  {12, 0, 2},
				  {13, 0, 3},
				  {14, 0, 4},
				  {0, 15, 5},
				  {0, 16, 6},
				  {0, 17, 0},
				  {0, 18, 8}};	*/
		
		FeatureSimilarity ls = new FeatureSimilarity(new DenseDoubleMatrix2D(feature));
		int[] indices = null;
		indices = ls.rankFeatures("Euclidean");
		System.out.println(indices.toString());
	}
}


