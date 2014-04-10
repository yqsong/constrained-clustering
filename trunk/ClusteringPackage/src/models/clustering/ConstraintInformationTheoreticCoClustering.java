package models.clustering;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import models.datastructure.ColtDenseVector;
import models.datastructure.ColtSparseVector;
import models.util.matrix.Matrix2DUtil;
import cern.colt.list.DoubleArrayList;
import cern.colt.list.IntArrayList;
import cern.colt.map.AbstractIntDoubleMap;
import cern.colt.map.OpenIntDoubleHashMap;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.jet.random.Uniform;

public class ConstraintInformationTheoreticCoClustering extends
		InformationTheoreticCoClustering {
	
	private List<AbstractIntDoubleMap> rowMustLinkConstrains;
	private List<AbstractIntDoubleMap> rowCannotLinkConstrains;
	private List<AbstractIntDoubleMap> columnMustLinkConstrains;
	private List<AbstractIntDoubleMap> columnCannotLinkConstrains;
	private int[][] rowMustlink = null;
	private int[][] rowCannotlink = null; 
	private int[][] columnMustlink = null;
	private int[][] columnCannotlink = null;
	
	private double[] rowMustWeight = null;
	private double[] rowCannotWeight = null;
	private double[] columnMustWeight = null;
	private double[] columnCannotWeight = null;
	
	private double MIN_DELTA_ICM = 0.001;
	private int MAX_ITER_ICM = 30;
	
	private boolean isIsingModel = false;

	public ConstraintInformationTheoreticCoClustering(double[][] dataMat, 
			int rowClusterNum, int colClusterNum, 
			int[][] rowMustlink, int[][] rowCannotlink, 
			int[][] columnMustlink, int[][] columnCannotlink) {
		super(dataMat, rowClusterNum, colClusterNum);
		this.rowMustlink = rowMustlink;
		this.rowCannotlink = rowCannotlink;
		this.columnMustlink = columnMustlink;
		this.columnCannotlink = columnCannotlink;

	}

	public ConstraintInformationTheoreticCoClustering(DoubleMatrix1D[] dataMat, 
			int rowClusterNum, int colClusterNum, 
			int[][] rowMustlink, int[][] rowCannotlink, 
			int[][] columnMustlink, int[][] columnCannotlink) {
		super(dataMat, rowClusterNum, colClusterNum);
		this.rowMustlink = rowMustlink;
		this.rowCannotlink = rowCannotlink;
		this.columnMustlink = columnMustlink;
		this.columnCannotlink = columnCannotlink;
	}

	public ConstraintInformationTheoreticCoClustering(List<DoubleMatrix1D> dataMat, 
			int rowClusterNum, int colClusterNum, 
			int[][] rowMustlink, int[][] rowCannotlink, 
			int[][] columnMustlink, int[][] columnCannotlink) {
		super(dataMat, rowClusterNum, colClusterNum);
		this.rowMustlink = rowMustlink;
		this.rowCannotlink = rowCannotlink;
		this.columnMustlink = columnMustlink;
		this.columnCannotlink = columnCannotlink;
	}

	public ConstraintInformationTheoreticCoClustering(double[][] dataMat, 
													int rowClusterNum, int colClusterNum, 
													int[][] rowMustlink, int[][] rowCannotlink, 
													int[][] columnMustlink, int[][] columnCannotlink,
													int seed) {
		super(dataMat, rowClusterNum, colClusterNum, seed);
		this.rowMustlink = rowMustlink;
		this.rowCannotlink = rowCannotlink;
		this.columnMustlink = columnMustlink;
		this.columnCannotlink = columnCannotlink;
	}
	
	public ConstraintInformationTheoreticCoClustering(DoubleMatrix1D[] dataMat, 
													int rowClusterNum, int colClusterNum, 
													int[][] rowMustlink, int[][] rowCannotlink, 
													int[][] columnMustlink, int[][] columnCannotlink,
													int seed) {
		super(dataMat, rowClusterNum, colClusterNum, seed);
		this.rowMustlink = rowMustlink;
		this.rowCannotlink = rowCannotlink;
		this.columnMustlink = columnMustlink;
		this.columnCannotlink = columnCannotlink;
	}
	
	public ConstraintInformationTheoreticCoClustering(List<DoubleMatrix1D> dataMat, 
													int rowClusterNum, int colClusterNum, 
													int[][] rowMustlink, int[][] rowCannotlink, 
													int[][] columnMustlink, int[][] columnCannotlink,
													int seed) {
		super(dataMat, rowClusterNum, colClusterNum, seed);
		this.rowMustlink = rowMustlink;
		this.rowCannotlink = rowCannotlink;
		this.columnMustlink = columnMustlink;
		this.columnCannotlink = columnCannotlink;
	}
	
protected void initialization() {
		
//		Timing time = new Timing();
		if (dtm == null) {
			System.err.println("Error: no data loaded!");
			return;
		}
		if (isDebug == true) {
			System.out.println("Infomation Theoretial Co-clustering. Initializaiton... ");
		}
		
// normalize data term matrix as a probabilistic matrix.
		
//		// data normalization
//		for (int i = 0; i < dtm.size(); ++i) {
//			double sum = 0.0;
//			IntArrayList indexList = new IntArrayList();
//			DoubleArrayList valueList = new DoubleArrayList();
//			dtm.get(i).getNonZeros(indexList, valueList);
//			for (int j = 0; j < indexList.size(); ++j) {
//				sum += valueList.get(j) * valueList.get(j);
//			}
//			sum = Math.sqrt(sum);
//			for (int j = 0; j < indexList.size(); ++j) {
//				int index = indexList.get(j);
//				double value = valueList.get(j);
//				dtm.get(i).set(index, value/(sum+Double.MIN_NORMAL));
//				DoubleMatrix1D vector = dtm.get(i);
//				vector = null;
//			}
//		}
		
		double sum = 0.0;
		for (int i = 0; i < rowNum; ++i) {
			IntArrayList indexList = new IntArrayList();
			DoubleArrayList valueList = new DoubleArrayList();
			dtm.get(i).getNonZeros(indexList, valueList);
			for (int j = 0; j < indexList.size(); ++j) {
				sum += valueList.get(j);
			}
		}
		for (int i = 0; i < rowNum; ++i) {
			IntArrayList indexList = new IntArrayList();
			DoubleArrayList valueList = new DoubleArrayList();
			DoubleMatrix1D vector = dtm.get(i);
			vector.getNonZeros(indexList, valueList);
			for (int j = 0; j < indexList.size(); ++j) {
				int index = indexList.get(j);
				dtm.get(i).setQuick(index, vector.getQuick(index)/sum);
			}
		}		
		
		qxy = new ArrayList<DoubleMatrix1D>();
		for (int i = 0; i < rowNum; ++i){
			qxy.add(new ColtSparseVector(columnNum));
		}
		for (int i = 0; i < rowNum; ++i) {
			for (int j = 0; j < columnNum; ++j) {
				double value = dtm.get(i).getQuick(j);
				if (value > 0) {
					qxy.get(i).setQuick(j, value);
				}
			}
			qxy.get(i).trimToSize();
		}
		qy_x = new ArrayList<DoubleMatrix1D>();
		qx_y = new ArrayList<DoubleMatrix1D>();
		for (int i = 0; i < columnNum; ++i){
			qy_x.add(new ColtSparseVector(rowNum));
		}
		for (int i = 0; i < rowNum; ++i){
			qx_y.add(new ColtSparseVector(columnNum));
		}

		if (isProfiling == true) {
			System.out.println("    Initialize qxy qx_y qy_x: ");
		}
		
		qxbyb = new DenseDoubleMatrix2D(rcNum, ccNum);
		
		qx_xb = new ArrayList<DoubleMatrix1D>();
		for (int i = 0; i < rowNum; ++i){
			qx_xb.add(new ColtDenseVector(rcNum));
		}
		qy_yb = new ArrayList<DoubleMatrix1D>();
		for (int i = 0; i < columnNum; ++i){
			qy_yb.add(new ColtDenseVector(ccNum));
		}
		
		if (isProfiling == true) {
			System.out.println("    Initialize qxbyb qx_xb qy_yb: ");
		}
		
		// initialize cluster membership matrix
		if (isDebug == true) {
			System.out.println("Initialize c1 c2...");
		}
		
		c1 = new ArrayList<DoubleMatrix1D>();
		for (int i = 0; i < rowNum; ++i){
			c1.add(new ColtDenseVector(rcNum));
		}
		c2 = new ArrayList<DoubleMatrix1D>();
		for (int i = 0; i < columnNum; ++i){
			c2.add(new ColtDenseVector(ccNum));
		}
		if (memberC1 == null) {
			if (isInitUseKmeans == true) {
				ConstraintKmeans kmeanscluster = new ConstraintKmeans(dtm, rcNum, "maxmin",
						this.rowMustlink, this.rowCannotlink, randomSeed);
				kmeanscluster.setDebug(isDebug);
				kmeanscluster.estimate();
				int[] labels = kmeanscluster.getLabels();
				memberC1 = new ColtDenseVector[2];
				memberC1[0] = new ColtDenseVector(labels.length);
				memberC1[1] = new ColtDenseVector(labels.length);
				for (int i = 0; i < labels.length; ++i) {
					memberC1[1].setQuick(i, labels[i]);
				}
			} else {
				Uniform randomGen = new Uniform(0, rcNum-1, randomSeed);
				memberC1 = new ColtDenseVector[2];
				memberC1[0] = new ColtDenseVector(rowNum);
				memberC1[1] = new ColtDenseVector(rowNum);
				for (int i = 0; i < memberC1[1].size(); ++i) {
					int value = randomGen.nextInt();
					if (value == rcNum) {
						value = rcNum - 1;
					}
					memberC1[1].setQuick(i, value);
				}
			}
		}
		assert (memberC1[1].size() == c1.size());
		for (int i = 0; i < c1.size(); ++i) {
			for (int j = 0; j < c1.get(i).size(); ++j) {
				if (j == memberC1[1].getQuick(i)) {
					c1.get(i).setQuick(j, 0);
				} else {
					c1.get(i).setQuick(j, 1);
				}
			}			
		}			
		if (isProfiling == true) {
			System.out.println("    Initialize c1: ");
		}
		
		if (memberC2 == null) {
			if (isInitUseKmeans == true) {
				List<DoubleMatrix1D> dtmTrans = new ArrayList<DoubleMatrix1D>();
				if (dtm.get(0) instanceof ColtDenseVector){
					for (int i = 0; i < columnNum; ++i) {
						dtmTrans.add(new ColtDenseVector(rowNum));
					}
				} else if (dtm.get(0) instanceof ColtSparseVector){
					for (int i = 0; i < columnNum; ++i) {
						dtmTrans.add(new ColtSparseVector(rowNum));
					}
				}
					
				for (int i = 0; i < rowNum; ++i) {
					for (int j = 0; j < dtm.get(i).size(); ++j) {
						if (dtm.get(i).getQuick(j) > 0) {
							dtmTrans.get(j).setQuick(i, dtm.get(i).getQuick(j));
						}
					}
				}
				
				ConstraintKmeans kmeanscluster = new ConstraintKmeans(dtmTrans, ccNum, "maxmin",
						this.columnMustlink, this.columnCannotlink, randomSeed);
//				GeneralKmeans kmeanscluster = new GeneralKmeans(dtmTrans, ccNum, "maxmin", randomSeed);
				kmeanscluster.setDebug(isDebug);
				kmeanscluster.estimate();
				int[] labels = kmeanscluster.getLabels();
				memberC2 = new ColtDenseVector[2];
				memberC2[0] = new ColtDenseVector(labels.length);
				memberC2[1] = new ColtDenseVector(labels.length);
				for (int i = 0; i < labels.length; ++i) {
					memberC2[1].setQuick(i, labels[i]);
				}
				dtmTrans = null;
			} else {
				Uniform randomGen = new Uniform(0, ccNum-1, randomSeed);
				memberC2 = new ColtDenseVector[2];
				memberC2[0] = new ColtDenseVector(columnNum);
				memberC2[1] = new ColtDenseVector(columnNum);
				for (int i = 0; i < memberC2[1].size(); ++i) {
					int value = randomGen.nextInt();
					if (value == ccNum) {
						value = ccNum-1;
					}
					memberC2[1].setQuick(i, value);
				}
			}
		} else {
			assert (memberC2[1].size() == c2.size());
		}
		for (int i = 0; i < c2.size(); ++i) {
			for (int j = 0; j < c2.get(i).size(); ++j) {
				if (j == memberC2[1].getQuick(i)) {
					c2.get(i).setQuick(j, 0);
				} else {
					c2.get(i).setQuick(j, 1);
				}
			}			
		}	
		
		if (isProfiling == true) {
			System.out.println("    Initialize c2: ");
		}
		
		qx = Matrix2DUtil.matrixSum(dtm, 2);
		qy = Matrix2DUtil.matrixSum(dtm, 1);
		
		for (int i = 0; i < rowNum; ++i) {
			IntArrayList indexList = new IntArrayList();
			DoubleArrayList valueList = new DoubleArrayList();
			DoubleMatrix1D vector = dtm.get(i);
			vector.getNonZeros(indexList, valueList);
			for (int j = 0; j < indexList.size(); ++j) {
				int index = indexList.get(j);
				double value = valueList.get(j);
				qy_x.get(index).setQuick(i, value/(qx.getQuick(i) + Double.MIN_VALUE));
				qx_y.get(i).setQuick(index, value/(qy.getQuick(index) + Double.MIN_VALUE));
			}
		}
		
		if (isProfiling == true) {
			System.out.println("    Initialize dtm qx qy qy_x qx_y: ");
		}
		
		update();
	
	}
	private void initializeConstriants() {
		//TODO: the speed is slow to compute all pairs...
		
//		if (this.rowMustWeight == null) {
//			this.mustWeight = new double[mustlink.length];
//			for (int i = 0; i < mustWeight.length; ++i) {
//				this.mustWeight[i] = 1/(Math.sqrt(data.size()) + Double.MIN_NORMAL);
//			}
//		} 
//		if (this.cannotWeight == null) {
//			this.cannotWeight = new double[cannotlink.length];
//			for (int i = 0; i < cannotWeight.length; ++i) {
//				this.cannotWeight[i] = 1/(Math.sqrt(data.size()) + Double.MIN_NORMAL);
//			}
//		}
//		
//		if (this.rowMustlink == null || this.rowMustlink.length == 0) {
//			this.rowMustWeight = 1;
//		} else {
//			this.rowMustWeight = 1/(Math.sqrt(rowMustlink.length) + Double.MIN_NORMAL);
//			this.rowMustWeight = 1/(Math.sqrt(this.rowNum) + Double.MIN_NORMAL);
//		}
//		
//		if (this.rowCannotlink == null || this.rowCannotlink.length == 0) {
//			this.rowCannotWeight = 1;
//		} else {
//			this.rowCannotWeight = 1/(Math.sqrt(rowMustlink.length) + Double.MIN_NORMAL);
//			this.rowCannotWeight = 1/(Math.sqrt(this.rowNum) + Double.MIN_NORMAL);
//		}
//		
//		if (this.columnMustlink == null || this.columnMustlink.length == 0) {
//			this.columnMustWeight = 1;
//		} else {
//			this.columnMustWeight = 1/(Math.sqrt(rowMustlink.length) + Double.MIN_NORMAL);
//			this.columnMustWeight = 1/(Math.sqrt(this.columnNum) + Double.MIN_NORMAL);
//		}
//		
//		if (this.columnCannotlink == null || this.columnCannotlink.length == 0) {
//			this.columnCannotWeight = 1;
//		} else {
//			this.columnCannotWeight = 1/(Math.sqrt(rowMustlink.length) + Double.MIN_NORMAL);
//			this.columnCannotWeight = 1/(Math.sqrt(this.columnNum) + Double.MIN_NORMAL);
//		}
		
		// for row pairs
		if (rowMustlink != null) {
			if (this.rowMustWeight == null) {
				this.rowMustWeight = new double[rowMustlink.length];
				for (int i = 0; i < rowMustWeight.length; ++i) {
					this.rowMustWeight[i] = 1/(Math.sqrt(this.rowNum) + Double.MIN_NORMAL);
//					this.rowMustWeight[i] = 1/((this.rowNum) + Double.MIN_NORMAL);
				}
			} 
			double maxValue = 0.0;
			double minValue = Double.MAX_VALUE;
			double sumValue = 0.0;
			List<AbstractIntDoubleMap> rowMustLinkWeights = new ArrayList<AbstractIntDoubleMap>();
			rowMustLinkConstrains = new ArrayList<AbstractIntDoubleMap>();
			for (int i = 0; i < rowNum; ++i) {
				AbstractIntDoubleMap hashmap = new OpenIntDoubleHashMap();
//				AbstractIntDoubleMap hashmap = new JavaIntDoubleHashMap();
				rowMustLinkConstrains.add(hashmap);
				
				AbstractIntDoubleMap hashmap1 = new OpenIntDoubleHashMap();
				rowMustLinkWeights.add(hashmap1);
			}
			for (int i = 0; i < rowMustlink.length; ++i) {
				int from = rowMustlink[i][0];
				int to = rowMustlink[i][1];
				
				if (i % 50000 == 0) {
					System.out.println("    Process row must link: " + i);
				}
				
				double sum = 0.0;
				if (this.isIsingModel == false) {
					for (int k = 0; k < qy_x.size(); ++k) {
						double temp = qy_x.get(k).getQuick(from);
						if (temp > 0.0) {
							sum += temp
							* ( Math.log( temp / 
							  ( qy_x.get(k).getQuick(to) + TOLERANCE ) ) / log2 );
						}
					}// end k
				} else {
					sum = 1;
				}
				
				if (maxValue < sum) {
					maxValue = sum;
				}
				if (minValue > sum) {
					minValue = sum;
				}
				sumValue += sum;
				
				if (rowMustLinkConstrains.get(from).containsKey(to) == false) {
					rowMustLinkConstrains.get(from).put(to, sum);
					rowMustLinkWeights.get(from).put(to, rowMustWeight[i]);
				}
				if (rowMustLinkConstrains.get(to).containsKey(from) == false) {
					rowMustLinkConstrains.get(to).put(from, sum);
					rowMustLinkWeights.get(to).put(from, rowMustWeight[i]);
				}
			}
			for (int i = 0; i < rowMustLinkConstrains.size(); ++i) {
				IntArrayList indexList = new IntArrayList();
				DoubleArrayList valueList = new DoubleArrayList();
				indexList = rowMustLinkConstrains.get(i).keys();
				valueList = rowMustLinkConstrains.get(i).values();
				
				for (int j = 0; j < indexList.size(); ++j) {
					int to = indexList.get(j);
					double value = valueList.get(j);
					if (isIsingModel == true) {
						value = sumValue/rowMustlink.length * rowMustLinkWeights.get(i).get(to);
//						value = sumValue/rowMustlink.length * 1/Math.sqrt(rowMustLinkWeights.get(i).size()) * rowMustLinkWeights.get(i).get(to);
					} else {
						value = value * rowMustLinkWeights.get(i).get(to);
					}
					rowMustLinkConstrains.get(i).put(to, value);
				}
			}
		}
		if (rowCannotlink != null) {
			if (this.rowCannotWeight == null) {
				this.rowCannotWeight = new double[rowCannotlink.length];
				for (int i = 0; i < rowCannotWeight.length; ++i) {
					this.rowCannotWeight[i] = 1/(Math.sqrt(this.rowNum) + Double.MIN_NORMAL);
//					this.rowCannotWeight[i] = 1/((this.rowNum) + Double.MIN_NORMAL);
				}
			} 
			double maxValue = 0.0;
			double minValue = Double.MAX_VALUE;
			double sumValue = 0.0;
			rowCannotLinkConstrains = new ArrayList<AbstractIntDoubleMap>();
			List<AbstractIntDoubleMap> cannotLinkWeights = new ArrayList<AbstractIntDoubleMap>();
			for (int i = 0; i < rowNum; ++i) {
				AbstractIntDoubleMap hashmap = new OpenIntDoubleHashMap();
//				AbstractIntDoubleMap hashmap = new JavaIntDoubleHashMap();
				rowCannotLinkConstrains.add(hashmap);
				
				AbstractIntDoubleMap hashmap1 = new OpenIntDoubleHashMap();
				cannotLinkWeights.add(hashmap1);
			}
			for (int i = 0; i < rowCannotlink.length; ++i) {
				int from = rowCannotlink[i][0];
				int to = rowCannotlink[i][1];

				if (i % 5000 == 0) {
					System.out.println("    Process row cannot link: " + i);
				}
				
				double sum = 0.0;
				if (this.isIsingModel == false) {
					for (int k = 0; k < qy_x.size(); ++k) {
						double temp = qy_x.get(k).getQuick(from);
						if (temp > 0.0) {
							sum += temp
							* ( Math.log( temp / 
							  ( qy_x.get(k).getQuick(to) + TOLERANCE ) ) / log2 );
						}
					}// end k
				} else {
					sum = 1;
				}
				

				if (maxValue < sum) {
					maxValue = sum;
				}
				if (minValue > sum) {
					minValue = sum;
				}
				sumValue += sum;

				if (rowCannotLinkConstrains.get(from).containsKey(to) == false) {
					rowCannotLinkConstrains.get(from).put(to, sum);
					cannotLinkWeights.get(from).put(to, rowCannotWeight[i]);
				}
				if (rowCannotLinkConstrains.get(to).containsKey(from) == false) {
					rowCannotLinkConstrains.get(to).put(from, sum);
					cannotLinkWeights.get(to).put(from, rowCannotWeight[i]);
				}
			}
			if (this.isIsingModel == true) {
				maxValue = 2;
			}
			// TODO: the max distance should be computed for all pairwise points..
			for (int i = 0; i < rowCannotLinkConstrains.size(); ++i) {
				IntArrayList indexList = new IntArrayList();
				DoubleArrayList valueList = new DoubleArrayList();
				indexList = rowCannotLinkConstrains.get(i).keys();
				valueList = rowCannotLinkConstrains.get(i).values();
				
				for (int j = 0; j < indexList.size(); ++j) {
					int to = indexList.get(j);
					double value = valueList.get(j);
					
					if (isIsingModel == true) {
						value = (maxValue - sumValue/rowCannotlink.length) * cannotLinkWeights.get(i).get(to);
					} else {
						value = (maxValue - value) * cannotLinkWeights.get(i).get(to);
					}
					rowCannotLinkConstrains.get(i).put(to, value);
				}
			}
		}
		
		// for column pairs
		if (columnMustlink != null) {
			if (this.columnMustWeight == null) {
				this.columnMustWeight = new double[columnMustlink.length];
				for (int i = 0; i < columnMustWeight.length; ++i) {
					this.columnMustWeight[i] = 1/(Math.sqrt(this.columnNum) + Double.MIN_NORMAL);
//					this.columnMustWeight[i] = 1;
				}
			} 

			double maxValue = 0.0;
			double minValue = Double.MAX_VALUE;
			double sumValue = 0.0;
			List<AbstractIntDoubleMap> columnMustLinkWeights = new ArrayList<AbstractIntDoubleMap>();
			columnMustLinkConstrains = new ArrayList<AbstractIntDoubleMap>();
			for (int i = 0; i < columnNum; ++i) {
				AbstractIntDoubleMap hashmap = new OpenIntDoubleHashMap();
//				AbstractIntDoubleMap hashmap = new JavaIntDoubleHashMap();
				columnMustLinkConstrains.add(hashmap);
				
				AbstractIntDoubleMap hashmap1 = new OpenIntDoubleHashMap();
				columnMustLinkWeights.add(hashmap1);
			}
			for (int i = 0; i < columnMustlink.length; ++i) {
				int from = columnMustlink[i][0];
				int to = columnMustlink[i][1];
				
				if (i % 5000 == 0) {
					System.out.println("    Process column must link: " + i);
				}
				
				double sum = 0.0;
				if (this.isIsingModel == false) {
					for (int k = 0; k < qx_y.size(); ++k) {
						double temp = qx_y.get(k).getQuick(from);
						if (temp > 0.0) {
							sum += temp
							* ( Math.log( temp / 
							  ( qx_y.get(k).getQuick(to) + TOLERANCE ) ) / log2 );
						}
					}// end k
				} else {
					sum = 1;
				}
				
				if (maxValue < sum) {
					maxValue = sum;
				}
				if (minValue > sum) {
					minValue = sum;
				}
				sumValue += sum;
				
				if (columnMustLinkConstrains.get(from).containsKey(to) == false) {
					columnMustLinkConstrains.get(from).put(to, sum);
					columnMustLinkWeights.get(from).put(to, columnMustWeight[i]);
				}
				if (columnMustLinkConstrains.get(to).containsKey(from) == false) {
					columnMustLinkConstrains.get(to).put(from, sum);
					columnMustLinkWeights.get(to).put(from, columnMustWeight[i]);
				}
			}
			for (int i = 0; i < columnMustLinkConstrains.size(); ++i) {
				IntArrayList indexList = new IntArrayList();
				DoubleArrayList valueList = new DoubleArrayList();
				indexList = columnMustLinkConstrains.get(i).keys();
				valueList = columnMustLinkConstrains.get(i).values();
				
				for (int j = 0; j < indexList.size(); ++j) {
					int to = indexList.get(j);
					double value = valueList.get(j);
					if (isIsingModel == true) {
						value = sumValue/columnMustlink.length * columnMustLinkWeights.get(i).get(to);
					} else {
						value = value * columnMustLinkWeights.get(i).get(to);
					}
					columnMustLinkConstrains.get(i).put(to, value);
				}
			}
		}
		if (columnCannotlink != null) {
			if (this.columnCannotWeight == null) {
				this.columnCannotWeight = new double[columnCannotlink.length];
				for (int i = 0; i < columnCannotWeight.length; ++i) {
					this.columnCannotWeight[i] = 1/(Math.sqrt(this.columnNum) + Double.MIN_NORMAL);
//					this.columnCannotWeight[i] = 1;
				}
			} 

			double maxValue = 0.0;
			double minValue = Double.MAX_VALUE;
			double sumValue = 0.0;
			columnCannotLinkConstrains = new ArrayList<AbstractIntDoubleMap>();
			List<AbstractIntDoubleMap> cannotLinkWeights = new ArrayList<AbstractIntDoubleMap>();
			for (int i = 0; i < columnNum; ++i) {
				AbstractIntDoubleMap hashmap = new OpenIntDoubleHashMap();
//				AbstractIntDoubleMap hashmap = new JavaIntDoubleHashMap();
				columnCannotLinkConstrains.add(hashmap);
				
				AbstractIntDoubleMap hashmap1 = new OpenIntDoubleHashMap();
				cannotLinkWeights.add(hashmap1);
			}
			for (int i = 0; i < columnCannotlink.length; ++i) {
				int from = columnCannotlink[i][0];
				int to = columnCannotlink[i][1];
				
				if (i % 5000 == 0) {
					System.out.println("    Process column cannot link: " + i);
				}
				
				double sum = 0.0;
				if (this.isIsingModel == false) {
					for (int k = 0; k < qx_y.size(); ++k) {
						double temp = qx_y.get(k).getQuick(from);
						if (temp > 0.0) {
							sum += temp
							* ( Math.log( temp / 
							  ( qx_y.get(k).getQuick(to) + TOLERANCE ) ) / log2 );
						}
					}// end k
				} else {
					sum = 1;
				}

				if (maxValue < sum) {
					maxValue = sum;
				}
				if (minValue > sum) {
					minValue = sum;
				}
				sumValue += sum;
				
				if (columnCannotLinkConstrains.get(from).containsKey(to) == false) {
					columnCannotLinkConstrains.get(from).put(to, sum);
					cannotLinkWeights.get(from).put(to, columnCannotWeight[i]);
				}
				if (columnCannotLinkConstrains.get(to).containsKey(from) == false) {
					columnCannotLinkConstrains.get(to).put(from, sum);
					cannotLinkWeights.get(to).put(from, columnCannotWeight[i]);
				}
			}
			if (this.isIsingModel == true) {
				maxValue = 2;
			}
			// TODO: the max distance should be computed for all pairwise points..
			for (int i = 0; i < columnCannotLinkConstrains.size(); ++i) {
				IntArrayList indexList = new IntArrayList();
				DoubleArrayList valueList = new DoubleArrayList();
				indexList = columnCannotLinkConstrains.get(i).keys();
				valueList = columnCannotLinkConstrains.get(i).values();

				for (int j = 0; j < indexList.size(); ++j) {
					int to = indexList.get(j);
					double value = valueList.get(j);
					
					if (isIsingModel == true) {
						value = (maxValue - sumValue/columnCannotlink.length) * cannotLinkWeights.get(i).get(to);
					} else {
						value = (maxValue - value) * cannotLinkWeights.get(i).get(to);
					}
					columnCannotLinkConstrains.get(i).put(to, value);
				}
			}
		}
	}
	
	public void estimate() {
		
		System.out.println("Constraint Information Theoretic Co-Clustering.");
		
		if (dtm == null) {
			System.err.println("Error: no data loaded");
		}
//		Timing timing = new Timing();
	
		this.initialization();
		super.estimate_core();
		
		System.out.println("Constraint ITCC initialization:");

		initializeConstriants();

		System.out.println("Constraint ITCC initialization of constraints:");
		
//		Timing time = new Timing();
		double oldCost = Double.MAX_VALUE;
		double cost = 0.0;
		double delta = 0.0;
		for (int iter = 0; iter < MAX_ITER; ++iter) {
			
			// for the row cluster
			List<DoubleMatrix1D> qyb_xb = new ArrayList<DoubleMatrix1D>();
			for (int i = 0; i < qxbyb.columns(); ++i) {
				qyb_xb.add(new ColtDenseVector(qxbyb.rows()));
			}
			for (int i = 0; i < qxbyb.rows(); ++i) {
				for (int j = 0; j < qxbyb.columns(); ++j) {
					qyb_xb.get(j).setQuick(i, qxbyb.getQuick(i, j) / (qxb.getQuick(i) + Double.MIN_VALUE));
				}
			}
			if (isProfiling == true) {
				System.out.println("    Estimate qyb_xb: ");
			}
			List<DoubleMatrix1D> qy_xb = Matrix2DUtil.DenseMultDense(qy_yb, qyb_xb);
			// smooth the probability
			double[] columnSum = new double[qy_xb.get(0).size()];
			Arrays.fill(columnSum, 0.0);
			for (int i = 0; i < qy_xb.size(); ++i) {
				for (int j = 0; j < qy_xb.get(i).size(); ++j) {
					columnSum[j] += qy_xb.get(i).getQuick(j);
				}
			}
			for (int i = 0; i < qy_xb.size(); ++i) {
				for (int j = 0; j < qy_xb.get(i).size(); ++j) {
					qy_xb.get(i).setQuick(j, Math.log(( qy_xb.get(i).getQuick(j) + TOLERANCE ) 
							/ ( columnSum[j] + TOLERANCE * qy_xb.size() ) ) );
				}
			}
			if (isProfiling == true) {
				System.out.println("    Estimate qy_xb: ");
			}
			
			// find new cluster distance with samples
			for (int i = 0; i < c1.size(); ++i) {
				c1.get(i).assign(0.0);
			}
			
			for (int k = 0; k < qy_x.size(); ++k) {
				IntArrayList indexList = new IntArrayList();
				DoubleArrayList valueList = new DoubleArrayList();
				qy_x.get(k).getNonZeros(indexList, valueList);
				for (int mm = 0; mm < indexList.size(); ++mm) {
					int i = indexList.get(mm);
					double temp = qy_x.get(k).getQuick(i);
					for (int j = 0; j < c1.get(i).size(); ++j) {
						double sum = c1.get(i).getQuick(j);
						sum += temp
							* ( ( Math.log( temp ) - qy_xb.get(k).getQuick(j) ) / log2 );
						c1.get(i).setQuick(j, sum);
					}
				}
			} 
			if (isProfiling == true) {
				System.out.println("    Estimate c1: ");
			}
			// update cluster centers and variational parameters
			// ICM algorithm for MRF
//			if (rowMustLinkConstrains != null || rowCannotLinkConstrains != null) {
			if (true) {
				double deltaICM = Double.MAX_VALUE;
				double icmCost = 0;
				double oldICMCost = Double.MAX_VALUE;
				int iterICM = 0;
				while (deltaICM > MIN_DELTA_ICM && iterICM < MAX_ITER_ICM) {
					int rowConflict = 0;
					
					icmCost = 0;
					for (int i = 0; i < c1.size(); ++i) {
						double[] likelihood = c1.get(i).toArray();
						double[] costValue = new double[c1.get(i).size()];
						Arrays.fill(costValue, 0.0);

						int icmIndex = 0;
						double minICMValue = Double.MAX_VALUE;
						for (int j = 0; j < c1.get(i).size(); ++j) {
							costValue[j] += likelihood[j];
							// for must link
							IntArrayList indexList = new IntArrayList();
							DoubleArrayList valueList = new DoubleArrayList();
							
							if (rowMustLinkConstrains != null && rowMustLinkConstrains.size() > 0) {
								indexList = rowMustLinkConstrains.get(i).keys();
								valueList = rowMustLinkConstrains.get(i).values();
								for (int mm = 0; mm < indexList.size(); ++mm) {
									int index = indexList.get(mm);
									if (memberC1[1].getQuick(index) != j) {
										costValue[j] += (valueList.get(mm));
										rowConflict++;
									}
								}
							}
							// for cannot link
							if (rowCannotLinkConstrains != null && rowCannotLinkConstrains.size() > 0) {
								indexList.clear();
								valueList.clear();
								indexList = rowCannotLinkConstrains.get(i).keys();
								valueList = rowCannotLinkConstrains.get(i).values();
								for (int mm = 0; mm < indexList.size(); ++mm) {
									int index = indexList.get(mm);
									if (memberC1[1].getQuick(index) == j) {
										costValue[j] += (valueList.get(mm));
										rowConflict++;
									}
								}
							}
							if (minICMValue > costValue[j]) {
								minICMValue = costValue[j];
								icmIndex = j;
							}
						}// end j
						icmCost += minICMValue;
						memberC1[1].setQuick(i, icmIndex);
					}// end i		
					deltaICM = Math.abs(icmCost - oldICMCost) / (oldICMCost + Double.MIN_VALUE);
					oldICMCost = icmCost;
					iterICM++;
					System.out.println("   Row Conflict number " + rowConflict);
				}// end while
			} else {
				memberC1 = Matrix2DUtil.matrixMin(c1, 2);				
			}
			if (isProfiling == true) {
				System.out.println("    Estimate memberC1: ");
			}
			
			update();

			// for the column cluster
			List<DoubleMatrix1D> qxb_yb = new ArrayList<DoubleMatrix1D>();
			for (int i = 0; i < qxbyb.rows(); ++i) {
				qxb_yb.add(new ColtDenseVector(qxbyb.columns()));
			}
			for (int i = 0; i < qxbyb.rows(); ++i) {
				for (int j = 0; j < qxbyb.columns(); ++j) {
					qxb_yb.get(i).setQuick(j, qxbyb.getQuick(i, j) / (qyb.getQuick(j) + Double.MIN_VALUE));
				}
			}
			if (isProfiling == true) {
				System.out.println("    Estimate qxb_yb: ");
			}
			List<DoubleMatrix1D> qx_yb = Matrix2DUtil.DenseMultDense(qx_xb, qxb_yb);

			// smooth the probability
			columnSum = new double[qx_yb.get(0).size()];
			Arrays.fill(columnSum, 0.0);
			for (int i = 0; i < qx_yb.size(); ++i) {
				for (int j = 0; j < qx_yb.get(i).size(); ++j) {
					columnSum[j] += qx_yb.get(i).getQuick(j);
				}
			}
			for (int i = 0; i < qx_yb.size(); ++i) {
				for (int j = 0; j < qx_yb.get(i).size(); ++j) {
					qx_yb.get(i).setQuick(j, Math.log( ( qx_yb.get(i).getQuick(j) + TOLERANCE ) 
							/ ( columnSum[j] + TOLERANCE * qx_yb.size() ) ) );
				}
			}
			if (isProfiling == true) {
				System.out.println("    Estimate qx_yb: ");
			}
			// find new cluster distance with samples
			for (int i = 0; i < c2.size(); ++i) {
				c2.get(i).assign(0.0);
			}
			
			for (int k = 0; k < qx_y.size(); ++k) {
				IntArrayList indexList = new IntArrayList();
				DoubleArrayList valueList = new DoubleArrayList();
				qx_y.get(k).getNonZeros(indexList, valueList);
				for (int mm = 0; mm < indexList.size(); ++mm) {
					int i = indexList.get(mm);
					double temp = qx_y.get(k).getQuick(i);
					for (int j = 0; j < c2.get(i).size(); ++j) {
						double sum = c2.get(i).getQuick(j);
						sum += temp
							* ( ( Math.log( temp ) - qx_yb.get(k).getQuick(j) ) / log2 );
						c2.get(i).setQuick(j, sum);
					}
				}
			} 
			if (isProfiling == true) {
				System.out.println("    Estimate c2: ");
			}
			
			// update cluster centers and variational parameters
			// ICM algorithm for MRF
//			if (columnMustLinkConstrains != null || columnCannotLinkConstrains != null) {
			if (true) {
				double deltaICM = Double.MAX_VALUE;
				double icmCost = 0;
				double oldICMCost = Double.MAX_VALUE;
				int iterICM = 0;
				while (deltaICM > MIN_DELTA_ICM && iterICM < MAX_ITER_ICM) {
					int columnConflict = 0;
					icmCost = 0;
					for (int i = 0; i < c2.size(); ++i) {
						double[] likelihood = c2.get(i).toArray();
						double[] costValue = new double[c2.get(i).size()];
						Arrays.fill(costValue, 0.0);

						int icmIndex = 0;
						double minICMValue = Double.MAX_VALUE;
						for (int j = 0; j < c2.get(i).size(); ++j) {
							costValue[j] += likelihood[j];
							// for must link
							IntArrayList indexList = new IntArrayList();
							DoubleArrayList valueList = new DoubleArrayList();
							
							if (columnMustLinkConstrains != null && columnMustLinkConstrains.size() > 0) {
								indexList = columnMustLinkConstrains.get(i).keys();
								valueList = columnMustLinkConstrains.get(i).values();
								for (int mm = 0; mm < indexList.size(); ++mm) {
									int index = indexList.get(mm);
									if (memberC2[1].getQuick(index) != j) {
										costValue[j] += (valueList.get(mm));
										columnConflict++;
									}
								}
							}
							// for cannot link
							if (columnCannotLinkConstrains != null && columnCannotLinkConstrains.size() > 0) { 
								indexList.clear();
								valueList.clear();
								indexList = columnCannotLinkConstrains.get(i).keys();
								valueList = columnCannotLinkConstrains.get(i).values();
								for (int mm = 0; mm < indexList.size(); ++mm) {
									int index = indexList.get(mm);
									if (memberC2[1].getQuick(index) == j) {
										costValue[j] += (valueList.get(mm));
										columnConflict++;
									}
								}
							}
							if (minICMValue > costValue[j]) {
								minICMValue = costValue[j];
								icmIndex = j;
							}
						}// end j
						icmCost += minICMValue;
						memberC2[1].setQuick(i, icmIndex);
					}// end i		
					deltaICM = Math.abs(icmCost - oldICMCost) / (oldICMCost + Double.MIN_VALUE);
					oldICMCost = icmCost;
					iterICM++;
					System.out.println("   Column Conflict number " + columnConflict);
				}// end while
			} else {
				memberC2 = Matrix2DUtil.matrixMin(c2, 2);				
			}
			if (isProfiling == true) {
				System.out.println("    Estimate memberC2: ");
			}
			
			update();
			fullGC();
			
			// find totally cost
			cost = 0.0;
			for (int i = 0; i < rowNum; ++i) {
				for (int j = 0; j < columnNum; ++j) {
					double temp1 = dtm.get(i).getQuick(j);
					double temp2 = qxy.get(i).getQuick(j);
					if (temp1 > 0 && temp2 > 0) {
						cost += temp1
							* ( Math.log(temp1 / (temp2 + Double.MIN_VALUE))
							/ log2 );
					}
				}
			}
			delta = Math.abs(oldCost - cost) / oldCost;
			if (isProfiling == true) {
				System.out.println("    Estimate qxy and cost delta: ");
			}
			
			System.out.println(" Constraint Interaion " + iter + " deltaMeans = " + delta + ": ");
			
			if (delta < MIN_DELTA) {
//				if (isDebug == true) 
				{
					System.out.println("Constraint Infomation Theoretial Co-clustering. Finished!");
				}
				break;
			}
			oldCost = cost;
		}// end iteration
			
	}
	
	public double[] getRowMustWeight() {
		return rowMustWeight;
	}

	public void setRowMustWeight(double[] rowMustWeight) {
		this.rowMustWeight = new double[rowMustWeight.length];
		this.rowMustWeight = Arrays.copyOf(rowMustWeight, rowMustWeight.length);
	}

	public double[] getRowCannotWeight() {
		return rowCannotWeight;
	}

	public void setRowCannotWeight(double[] rowCannotWeight) {
		this.rowCannotWeight = new double[rowCannotWeight.length];
		this.rowCannotWeight = Arrays.copyOf(rowCannotWeight, rowCannotWeight.length);
	}

	public double[] getColumnMustWeight() {
		return columnMustWeight;
	}

	public void setColumnMustWeight(double[] columnMustWeight) {
		this.columnMustWeight = new double[columnMustWeight.length];
		this.columnMustWeight = Arrays.copyOf(columnMustWeight, columnMustWeight.length);
	}

	public double[] getColumnCannotWeight() {
		return columnCannotWeight;
	}

	public void setColumnCannotWeight(double[] columnCannotWeight) {
		this.columnCannotWeight = new double[columnCannotWeight.length];
		this.columnCannotWeight = Arrays.copyOf(columnCannotWeight, columnCannotWeight.length);	
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
				  {0.03, 0.05, 0.05, 0,    0,    0},
				  {0.05, 0.03, 0.05, 0,    0,    0},
				  {0.01, 0.01, 0.01, 0.05, 0.04, 0.05},
				  {0.04, 0.04, 0.04, 0.05, 0.05, 0.04},
				  {0.04, 0.04, 0.03, 0.03, 0.04, 0.04},
				  {0.03, 0.04, 0.03, 0.03, 0.04, 0.03},
				  };	
		int[][] rowMust = {
				{0, 1},	
				{2, 3},	
				{4, 5},	
		};
		int[][] rowCannot = {
				{1, 2},	
				{3, 5},	
		};
		int[][] colMust = {
				{0, 1},	
				{1, 2},	
				{3, 4},	
				{4, 5},
		};
		int[][] colCannot = {
				{2, 3},	
		};
		InformationTheoreticCoClustering itcc = new InformationTheoreticCoClustering(feature, 3, 2);
		itcc.estimate();
		int[] rlabel = itcc.getRowClusterLabels();
		int[] clabel = itcc.getColumnClusterLabels();
		for (int i = 0; i < rlabel.length; ++i) {
			System.out.print(rlabel[i] + " ");
		}
		System.out.println("");
		for (int i = 0; i < rlabel.length; ++i) {
			System.out.print(clabel[i] + " ");
		}
		System.out.println("");
		
		ConstraintInformationTheoreticCoClustering citcc = 
			new ConstraintInformationTheoreticCoClustering(feature, 3, 2, rowMust, rowCannot, colMust, colCannot);
		citcc.setRandomSeed(11122345);
		citcc.estimate();
		rlabel = citcc.getRowClusterLabels();
		clabel = citcc.getColumnClusterLabels();
		for (int i = 0; i < rlabel.length; ++i) {
			System.out.print(rlabel[i] + " ");
		}
		System.out.println("");
		for (int i = 0; i < rlabel.length; ++i) {
			System.out.print(clabel[i] + " ");
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
