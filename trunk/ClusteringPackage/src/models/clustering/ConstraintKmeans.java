package models.clustering;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import models.datastructure.ColtDenseVector;
import models.datastructure.ColtSparseVector;
import cern.colt.list.DoubleArrayList;
import cern.colt.list.IntArrayList;
import cern.colt.map.AbstractIntDoubleMap;
import cern.colt.map.OpenIntDoubleHashMap;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.SparseDoubleMatrix1D;


public class ConstraintKmeans extends GeneralKmeans{

	private List<AbstractIntDoubleMap> mustLinkConstrains;
	private List<AbstractIntDoubleMap> cannotLinkConstrains;
	
	private int[][] mustlink = null;
	private int[][] cannotlink = null;
	
	private double[] mustWeight = null;
	private double[] cannotWeight = null;
	
	private double MIN_DELTA_ICM = 0.001;
	private int MAX_ITER_ICM = 30;
	
	private boolean isIsingModel = false;
	private boolean testSimilarity = false;

	public ConstraintKmeans(double[][] data, int cNum, int[][] mustlink, int[][] cannotlink) {
		this(data, false, cNum, "maxmin", mustlink, cannotlink, 0);
	}
	
	public ConstraintKmeans(DoubleMatrix1D[] data, int cNum, int[][] mustlink, int[][] cannotlink) {
		this(data, cNum, "maxmin", mustlink, cannotlink, 0);
	}

	public ConstraintKmeans(List<DoubleMatrix1D> data, int cNum, int[][] mustlink, int[][] cannotlink) {
		this(data, cNum, "maxmin", mustlink, cannotlink, 0);
	}

	public ConstraintKmeans(double[][] data, int cNum, int[][] mustlink, int[][] cannotlink, int seed) {
		this(data, false, cNum, "maxmin", mustlink, cannotlink, seed);
	}
	
	public ConstraintKmeans(DoubleMatrix1D[] data, int cNum, int[][] mustlink, int[][] cannotlink, int seed) {
		this(data, cNum, "maxmin", mustlink, cannotlink, seed);
	}

	public ConstraintKmeans(List<DoubleMatrix1D> data, int cNum, int[][] mustlink, int[][] cannotlink, int seed) {
		this(data, cNum, "maxmin", mustlink, cannotlink, seed);
	}

	public ConstraintKmeans(double[][] dataMat, boolean isDense, int cNum, String method, int[][] mustlink, int[][] cannotlink, int seed) {
		super(dataMat, isDense, cNum, method, seed);
		this.mustlink = mustlink;
		this.cannotlink = cannotlink;
	}
	
	public ConstraintKmeans(DoubleMatrix1D[] dataMat, int cNum, String method, int[][] mustlink, int[][] cannotlink, int seed) {
		super(dataMat, cNum, method, seed);
		this.mustlink = mustlink;
		this.cannotlink = cannotlink;
	}
	
	public ConstraintKmeans(List<DoubleMatrix1D> dataMat, int cNum, String method, int[][] mustlink, int[][] cannotlink, int seed) {
		super(dataMat, cNum, method, seed);
		this.mustlink = mustlink;
		this.cannotlink = cannotlink;
	}
	
	private void initializeConstriants() {
		
		if (testSimilarity == true) {
			testSimilarity ();
		}
		
		if (mustlink != null) {
			
			if (this.mustWeight == null) {
				this.mustWeight = new double[mustlink.length];
				for (int i = 0; i < mustWeight.length; ++i) {
					this.mustWeight[i] = 1/(Math.sqrt(data.size()) + Double.MIN_NORMAL);
//					this.mustWeight[i] = 1/((data.size()) + Double.MIN_NORMAL);
				}
			} 
			
			double maxValue = 0.0;
			double minValue = Double.MAX_VALUE;
			double sumValue = 0.0;
			List<AbstractIntDoubleMap> mustLinkWeights = new ArrayList<AbstractIntDoubleMap>();
			mustLinkConstrains = new ArrayList<AbstractIntDoubleMap>();
			for (int i = 0; i < data.size(); ++i) {
				AbstractIntDoubleMap hashmap = new OpenIntDoubleHashMap();
//				AbstractIntDoubleMap hashmap = new JavaIntDoubleHashMap();
				mustLinkConstrains.add(hashmap);
				
				AbstractIntDoubleMap hashmap1 = new OpenIntDoubleHashMap();
				mustLinkWeights.add(hashmap1);
			}
			for (int i = 0; i < mustlink.length; ++i) {
				int from = mustlink[i][0];
				int to = mustlink[i][1];
				
				if (i % 50000 == 0) {
					System.out.println("    Process row must link: " + i);
				}
				
				double instDist = 0.0;
				if (this.isIsingModel == false) {
					if (data.get(from) instanceof ColtSparseVector)
						instDist = computeDistance(data.get(from), data.get(to), 
								((ColtSparseVector) data.get(from)).getNormValue(), 
								((ColtSparseVector) data.get(to)).getNormValue());
					if (data.get(from) instanceof ColtDenseVector)
						instDist = computeDistance(data.get(from), data.get(to), 
								((ColtDenseVector) data.get(from)).getNormValue(), 
								((ColtDenseVector) data.get(to)).getNormValue());
				} else {
					instDist = 1;
				}
				
				if (maxValue < instDist) {
					maxValue = instDist;
				}
				if (minValue > instDist) {
					minValue = instDist;
				}
				sumValue += instDist;
				
				if (mustLinkConstrains.get(from).containsKey(to) == false) {
					mustLinkConstrains.get(from).put(to, instDist);
					mustLinkWeights.get(from).put(to, mustWeight[i]);
				}
				if (mustLinkConstrains.get(to).containsKey(from) == false) {
					mustLinkConstrains.get(to).put(from, instDist);
					mustLinkWeights.get(to).put(from, mustWeight[i]);
				}
			}
			for (int i = 0; i < mustLinkConstrains.size(); ++i) {
				IntArrayList indexList = new IntArrayList();
				DoubleArrayList valueList = new DoubleArrayList();
				indexList = mustLinkConstrains.get(i).keys();
				valueList = mustLinkConstrains.get(i).values();
				
				for (int j = 0; j < indexList.size(); ++j) {
					int to = indexList.get(j);
					double value = valueList.get(j);
					if (isIsingModel == true) {
						value = sumValue/mustlink.length * mustLinkWeights.get(i).get(to);
//						value = sumValue/mustlink.length * 1/Math.sqrt(mustLinkWeights.get(i).size()) * mustLinkWeights.get(i).get(to);
					} else {
						value = value * mustLinkWeights.get(i).get(to);
					}
					mustLinkConstrains.get(i).put(to, value);
				}
			}
		}
		if (cannotlink != null) {
			if (this.cannotWeight == null) {
				this.cannotWeight = new double[cannotlink.length];
				for (int i = 0; i < cannotWeight.length; ++i) {
					this.cannotWeight[i] = 1/(Math.sqrt(data.size()) + Double.MIN_NORMAL);
//					this.cannotWeight[i] = 1/((data.size()) + Double.MIN_NORMAL);
				}
			}
			
			double maxValue = 0.0;
			double minValue = Double.MAX_VALUE;
			double sumValue = 0.0;
			List<AbstractIntDoubleMap> cannotLinkWeights = new ArrayList<AbstractIntDoubleMap>();
			cannotLinkConstrains = new ArrayList<AbstractIntDoubleMap>();
			for (int i = 0; i < data.size(); ++i) {
				AbstractIntDoubleMap hashmap = new OpenIntDoubleHashMap();
//				AbstractIntDoubleMap hashmap = new JavaIntDoubleHashMap();
				cannotLinkConstrains.add(hashmap);
				
				AbstractIntDoubleMap hashmap1 = new OpenIntDoubleHashMap();
				cannotLinkWeights.add(hashmap1);
			}
			for (int i = 0; i < cannotlink.length; ++i) {
				int from = cannotlink[i][0];
				int to = cannotlink[i][1];
				
				if (i % 5000 == 0) {
					System.out.println("    Process row cannot link: " + i);
				}
				
				double instDist = 0.0;
				if (this.isIsingModel == false) {
					if (data.get(from) instanceof ColtSparseVector)
						instDist = computeDistance(data.get(from), data.get(to), 
								((ColtSparseVector) data.get(from)).getNormValue(), 
								((ColtSparseVector) data.get(to)).getNormValue());
					if (data.get(from) instanceof ColtDenseVector)
						instDist = computeDistance(data.get(from), data.get(to), 
								((ColtDenseVector) data.get(from)).getNormValue(), 
								((ColtDenseVector) data.get(to)).getNormValue());
				} else {
					instDist = 1;
				}

				if (maxValue < instDist) {
					maxValue = instDist;
				}
				if (minValue > instDist) {
					minValue = instDist;
				}
				sumValue += instDist;
				if (cannotLinkConstrains.get(from).containsKey(to) == false) {
					cannotLinkConstrains.get(from).put(to, instDist);
					cannotLinkWeights.get(from).put(to, cannotWeight[i]);
				}
				if (cannotLinkConstrains.get(to).containsKey(from) == false) {
					cannotLinkConstrains.get(to).put(from, instDist);
					cannotLinkWeights.get(to).put(from, cannotWeight[i]);
				}
			}
			if (this.isIsingModel == true) {
				maxValue = 2;
			}
			// TODO: the max distance should be computed for all pairwise points..
			for (int i = 0; i < cannotLinkConstrains.size(); ++i) {
				IntArrayList indexList = new IntArrayList();
				DoubleArrayList valueList = new DoubleArrayList();
				indexList = cannotLinkConstrains.get(i).keys();
				valueList = cannotLinkConstrains.get(i).values();
				
				for (int j = 0; j < indexList.size(); ++j) {
					int to = indexList.get(j);
					double value = valueList.get(j);
					if (isIsingModel == true) {
						value = (maxValue - sumValue/cannotlink.length) * cannotLinkWeights.get(i).get(to);
					} else {
						value = (maxValue - value) * cannotLinkWeights.get(i).get(to);
					}
					cannotLinkConstrains.get(i).put(to, value);
				}
			}
		}
	}
	
	public double[] getMustWeight() {
		return mustWeight;
	}

	public void setMustWeight(double[] mustWeight) {
		this.mustWeight = new double[mustWeight.length];
		this.mustWeight = Arrays.copyOf(mustWeight, mustWeight.length);
	}

	public double[] getCannotWeight() {
		return cannotWeight;
	}

	public void setCannotWeight(double[] cannotWeight) {
		this.cannotWeight = new double[cannotWeight.length];
		this.cannotWeight = Arrays.copyOf(cannotWeight, cannotWeight.length);
	}

	public void testSimilarity () {
		double [][] similarity = new double [data.size()][data.size()];
		double meanAll = 0;
		double varAll = 0;
		for (int i = 0; i < data.size(); ++i) {
			for (int j = 0; j < data.size(); ++j) {
				if (data.get(i) instanceof ColtSparseVector)
					similarity[i][j] = 1 - computeDistance(data.get(i), data.get(j), 
							((ColtSparseVector) data.get(i)).getNormValue(), 
							((ColtSparseVector) data.get(j)).getNormValue());
				if (data.get(i) instanceof ColtDenseVector)
					similarity[i][j] = 1 - computeDistance(data.get(i), data.get(j), 
							((ColtDenseVector) data.get(i)).getNormValue(), 
							((ColtDenseVector) data.get(j)).getNormValue());
				meanAll += similarity[i][j];
			}
		}
		meanAll /= (data.size() * data.size());
		for (int i = 0; i < data.size(); ++i) {
			for (int j = 0; j < data.size(); ++j) {
				varAll += (similarity[i][j] - meanAll) * (similarity[i][j] - meanAll);
			}
		}
		varAll = Math.sqrt(varAll/(data.size() - 1));
		
		double meanMustLink = 0;
		double varMustLink = 0;
		if (mustlink != null) {
			for (int i = 0; i < mustlink.length; ++i) {
				meanMustLink += similarity[mustlink[i][0]][mustlink[i][1]];
			}
			meanMustLink /= mustlink.length;
			for (int i = 0; i < mustlink.length; ++i) {
				varMustLink += (similarity[mustlink[i][0]][mustlink[i][1]] - meanMustLink) 
				* (similarity[mustlink[i][0]][mustlink[i][1]] - meanMustLink);
			}
			varMustLink = Math.sqrt(varMustLink/(data.size() - 1));
		}
		
		System.out.println("Mean All: " + meanAll + " and Var all: " + varAll);
		System.out.println("Mean MustLink: " + meanMustLink + " and Var MustLink: " + varMustLink);
	}
	
	public void estimate() {
		// Initialize clusterMeans
//		Timing timing = new Timing();

//		Timing profiling = new Timing();

		if (isDebug == true) {
			System.out.println("Entering Constraint KMeans initialization..");
		}
		
		// initialization centers
//		if(this.initMethod.equalsIgnoreCase("orthogonal")) {
//			initializeOrthogonalMeansSample();
//		} else if (initMethod.equalsIgnoreCase("maxmin")) {
//			initializeMeansSample();
//		} else {
//			initializeRandomMeansSample();
//		}
		super.estimate();
		
		if (isDebug == true) {
			System.out.println("Constraint Kmeans initialization:");
		}		
		
		initializeConstriants();
		if (isDebug == true) {
			System.out.println("Constraint Kmeans initialization of constriants:");
		}		
		
		List<List<DoubleMatrix1D>> instanceClusters = 
			new ArrayList<List<DoubleMatrix1D>>(clusterNum);
		for (int c = 0; c < clusterNum; c++) {
			instanceClusters.add(c, new ArrayList<DoubleMatrix1D>());
		}
		
		double deltaMeans = 1.0;
		double deltaPoints = data.size();
		double cost = 1.0;
		double oldCost = Double.MAX_VALUE;
		double deltaCost = Double.MAX_VALUE;
		int iterations = 0;

		if (isDebug == true) {
			System.out.println("Entering Constraint KMeans iteration..");
		}
		
		
		double[][] likelihoodMatrix = new double[data.size()][clusterNum];
		double instCenterDist = 0;
		ArrayList<Double> centerNorm = new ArrayList<Double>();
//		for (int i = 0; i < centers.size(); ++i) {
//			double norm = 0.0;
//			DoubleMatrix1D v1 = centers.get(i);
//			if (v1 instanceof SparseDoubleMatrix1D) {
//				norm = normQuick((SparseDoubleMatrix1D) v1);
//			} else {
//				norm = product(v1, v1);
//			}
//			centerNorm.add(norm);
//		}
//		cost = 0.0;
//		for (int n = 0; n < data.size(); n++) {
//			int minIndex = 0;
//			double minValue = Double.MAX_VALUE;
//			for (int c = 0; c < clusterNum; c++) {
//				if (data.get(n) instanceof ColtSparseVector)
//					instCenterDist = computeDistance(data.get(n), centers.get(c), 
//							((ColtSparseVector) data.get(n)).getNormValue(), centerNorm.get(c));
//				if (data.get(n) instanceof ColtDenseVector)
//					instCenterDist = computeDistance(data.get(n), centers.get(c),
//							((ColtDenseVector) data.get(n)).getNormValue(), centerNorm.get(c));
////				instCenterDist = computeDistance(centers.get(c), data.get(n), centerNorm.get(c), 
////						((ColtDenseVector) data.get(n)).getNormValue());
//				if (minValue > instCenterDist) {
//					minValue = instCenterDist;
//					minIndex = c;
//				}
//			}
//			clusterLabels[n] = minIndex;
//			cost += instCenterDist;
//			instanceClusters.get(minIndex).add(data.get(n));
//		}
//		deltaCost = Math.abs(cost - oldCost) / (oldCost + Double.MIN_VALUE);
//		oldCost = cost;
//		deltaMeans = update(instanceClusters);
//		iterations++;
//		if (isDebug == true) {
////			System.out.println(" Iter " + iterations + " deltaMeans = " + deltaMeans);
//			System.out.println(" Constraint Interaion " + iterations
//					+ " deltaCost = " + deltaCost
//					+ " deltaMeans = " + deltaMeans + ": ");
//		}
		
		
		while (deltaCost > COST_CHANGE_DELTA &&
				deltaMeans > MEANS_TOLERANCE && iterations < MAX_ITER
				&& deltaPoints > data.size() * POINTS_TOLERANCE) {

			iterations++;
			deltaPoints = 0;

			centerNorm = new ArrayList<Double>();
			for (int i = 0; i < centers.size(); ++i) {
				double norm = 0.0;
				DoubleMatrix1D v1 = centers.get(i);
				if (v1 instanceof SparseDoubleMatrix1D) {
					norm = normQuick((SparseDoubleMatrix1D) v1);
				} else {
					norm = product(v1, v1);
				}
				centerNorm.add(norm);
			}
			
			for (int n = 0; n < data.size(); n++) {
				for (int c = 0; c < clusterNum; c++) {
					if (data.get(n) instanceof ColtSparseVector)
						instCenterDist = computeDistance(data.get(n), centers.get(c), 
								((ColtSparseVector) data.get(n)).getNormValue(), centerNorm.get(c));
					if (data.get(n) instanceof ColtDenseVector)
						instCenterDist = computeDistance(data.get(n), centers.get(c),
								((ColtDenseVector) data.get(n)).getNormValue(), centerNorm.get(c));
//					instCenterDist = computeDistance(centers.get(c), data.get(n), centerNorm.get(c), 
//							((ColtDenseVector) data.get(n)).getNormValue());
					likelihoodMatrix[n][c] = instCenterDist;
				}
			}
			
			double icmCost = 1.0;
			double oldICMCost = Double.MAX_VALUE;
			double deltaICM = Double.MAX_VALUE;
			int iterICM = 0;
			while (deltaICM > MIN_DELTA_ICM && iterICM < MAX_ITER_ICM) {
				int conflictNum = 0;
				
				icmCost = 0;
//				System.out.println("icm iteration: " + iterICM);
				// i just as n
				int[] newClusterLabels = new int[clusterLabels.length];
				for (int n = 0; n < likelihoodMatrix.length; ++n) {
					double[] likelihood = likelihoodMatrix[n];
					double[] costValue = new double[likelihood.length];
					Arrays.fill(costValue, 0.0);

					int icmIndex = 0;
					double minICMValue = Double.MAX_VALUE;

					// j just as c
					for (int c = 0; c < likelihood.length; ++c) {
						costValue[c] += likelihood[c];
						// for must link
						IntArrayList indexList = new IntArrayList();
						DoubleArrayList valueList = new DoubleArrayList();
						
						if (mustLinkConstrains != null && mustLinkConstrains.size() > 0) {
							indexList = mustLinkConstrains.get(n).keys();
							valueList = mustLinkConstrains.get(n).values();
							for (int mm = 0; mm < indexList.size(); ++mm) {
								int index = indexList.get(mm);
								if (clusterLabels[index] != c) {
									conflictNum++;
									costValue[c] += (valueList.get(mm));
								} 
							}
						}
						// for cannot link
						if (cannotLinkConstrains != null && cannotLinkConstrains.size() > 0) {
							indexList.clear();
							valueList.clear();
							indexList = cannotLinkConstrains.get(n).keys();
							valueList = cannotLinkConstrains.get(n).values();
							for (int mm = 0; mm < indexList.size(); ++mm) {
								int index = indexList.get(mm);
								if (clusterLabels[index] == c) {
									conflictNum++;
									costValue[c] += (valueList.get(mm));
								}
							}
						}
						if (minICMValue > costValue[c]) {
							minICMValue = costValue[c];
							icmIndex = c;
						}
					}// end j
					icmCost += minICMValue;
					newClusterLabels[n] = icmIndex;
					if (clusterLabels[n] != icmIndex) {
						clusterLabels[n] = icmIndex;
						deltaPoints++;
					} 
				}// end i	
//				clusterLabels = newClusterLabels;
				deltaICM = Math.abs(icmCost - oldICMCost) / (oldICMCost + Double.MIN_VALUE);
				oldICMCost = icmCost;
				iterICM++;
				System.out.println("   Conflict number " + conflictNum);
			}// end while
			
			cost = icmCost;
			deltaCost = Math.abs(cost - oldCost) / (oldCost + Double.MIN_VALUE);
			oldCost = cost;

			
			// Add to closest cluster & label it such
			for (int n = 0; n < clusterLabels.length; ++n) {
				instanceClusters.get(clusterLabels[n]).add(data.get(n));

			}
			
			if (isProfiling == true) {
				System.out.println("    Profiling: Compute distances from each point to centers:");
			}
			

			deltaMeans = update(instanceClusters);
			
			if (isDebug == true && deltaMeans == -1) {
				System.out.println("Can't find an instance to move.  Exiting.");
			}

			if (isProfiling == true) {
				System.out.println("    Profiling: Compute center means: ");
			}
			
			if (isDebug == true) {
//				System.out.println(" Iter " + iterations + " deltaMeans = " + deltaMeans);
				System.out.println(" Constraint Interaion " + iterations
						+ " deltaCost = " + deltaCost
						+ " deltaMeans = " + deltaMeans + ": ");
			}
		}

		if (deltaCost <= COST_CHANGE_DELTA) {
			if (isDebug == true) 
			{
				System.out.println("Constraint KMeans converged with deltaCost = " + deltaCost);
			}
		}
		if (deltaMeans <= MEANS_TOLERANCE) {
			if (isDebug == true) 
			{
				System.out.println("Constraint KMeans converged with deltaMeans = " + deltaMeans);
			}
		}
		else if (iterations >= MAX_ITER) {
			if (isDebug == true) 
			{
				System.out.println("Constraint KMeans Maximum number of iterations (" + MAX_ITER + ") reached.");
			}
		}
		else if (deltaPoints <= data.size() * POINTS_TOLERANCE) {
			if (isDebug == true) 
			{
				System.out.println("Constraint KMeans Minimum number of points (np*" + POINTS_TOLERANCE + "="
					+ (int) (data.size() * POINTS_TOLERANCE)
					+ ") moved in last iteration. Saying converged.");
			}
		}
	}
	
	// mini test
	public static void main (String[] args) {

//		double[][] feature = {
//				  {0.03, 0.05, 0.05, 0, 0, 0},
//				  {0.05, 0.03, 0.05, 0, 0, 0},
//				  {0, 0, 0, 0.05, 0.04, 0.05},
//				  {0, 0, 0, 0.05, 0.05, 0.04},
//				  {0.04, 0.04, 0, 0.03, 0.04, 0.04},
//				  {0.03, 0.04, 0.03, 0, 0.04, 0.03},
//				  };	
		
		double[][] feature = {
				  {0.03, 0.05, 0.05, 0, 0, 0},
				  {0.05, 0.03, 0.05, 0, 0, 0},
				  {0.03, 0.03, 0.03, 0.05, 0.04, 0.05},
				  {0, 0, 0, 0.05, 0.05, 0.04},
				  {0.04, 0.04, 0.03, 0.03, 0.04, 0.04},
				  {0.03, 0.04, 0.03, 0.03, 0.04, 0.03},
				  };	
		int[][] rowMust = {
				{0, 1},	
				{2, 3},	
				{4, 5},	
		};
		int[][] rowCannot = {
				{2, 4},	
				{2, 5},	
		};
		ConstraintKmeans ckmeans = 
			new ConstraintKmeans(feature, true, 3, "maxmin", rowMust, rowCannot, 0);
		ckmeans.setRandomSeed(11122345);
		ckmeans.estimate();
		int[] label = ckmeans.getLabels();
		for (int i = 0; i < label.length; ++i) {
			System.out.print(label[i] + " ");
		}
		
		System.out.println("");
	}

	public boolean isIsingModel() {
		return isIsingModel;
	}

	public void setIsingModel(boolean isIsingModel) {
		this.isIsingModel = isIsingModel;
	}
}
