package models.clustering;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import models.datastructure.ColtDenseVector;
import models.datastructure.ColtSparseVector;
import models.util.matrix.Matrix2DUtil;
import cern.colt.list.DoubleArrayList;
import cern.colt.list.IntArrayList;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.jet.random.Uniform;

/**
 * Information-Theoretic Co-Clustering
 * 
 * See
 @inproceedings{Dhillon03,
	author = {Dhillon, I. S.  and Mallela, S.  and Modha, D. S. },
	booktitle = {Proceedings of The Ninth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)},
	pages = {89--98},
	title = {Information-Theoretic Co-Clustering},
	year = {2003}
 }
 *  
 * @author Yangqiu Song
 */

// implemented with 1D sparse arrays
public class InformationTheoreticCoClustering {
	protected int MAX_ITER = 100;
	protected double MIN_DELTA = 0.001;
	protected double TOLERANCE = 0.00000000000000000001;
	
	protected boolean isDebug = true;
	
	protected boolean isProfiling = false;
	
	protected boolean isInitUseKmeans = true;

	// matrix row and column number
	protected int rowNum;
	protected int columnNum;
	// cluster number
	protected int rcNum;
	protected int ccNum;
	
	// data term matrix
	protected List<DoubleMatrix1D> dtm = null;

	protected List<DoubleMatrix1D> c1 = null;
	protected List<DoubleMatrix1D> c2 = null;

	protected DoubleMatrix1D[] memberC1 = null;
	protected DoubleMatrix1D[] memberC2 = null;
	
	protected DoubleMatrix1D qx = null;
	protected DoubleMatrix1D qy = null;
	
	protected DoubleMatrix2D qxbyb = null;

	protected DoubleMatrix1D qxb = null;
	protected DoubleMatrix1D qyb = null;
	
	protected List<DoubleMatrix1D> qx_xb = null;
	protected List<DoubleMatrix1D> qy_yb = null;
	

	protected List<DoubleMatrix1D> qxy = null;
	
	protected List<DoubleMatrix1D> qy_x = null;
	protected List<DoubleMatrix1D> qx_y = null;
	
	protected int randomSeed = 0;
	protected Uniform random = null;
	
	protected final double log2 = Math.log(2);

	public InformationTheoreticCoClustering(double[][] dataMat, int rowClusterNum, int colClusterNum) {
		this(dataMat, rowClusterNum, colClusterNum, 0);
	}
	public InformationTheoreticCoClustering(DoubleMatrix1D[] dataMat, int rowClusterNum, int colClusterNum) {
		this(dataMat, rowClusterNum, colClusterNum, 0);
	}
	public InformationTheoreticCoClustering(List<DoubleMatrix1D> dataMat, int rowClusterNum, int colClusterNum) {
		this(dataMat, rowClusterNum, colClusterNum, 0);
	}

	public InformationTheoreticCoClustering(int rowNum, int columnNum, int rowClusterNum, int colClusterNum, int seed) {
		this.rcNum = rowClusterNum;
		this.ccNum = colClusterNum;
		this.rowNum = rowNum;
		this.columnNum = columnNum;
		this.randomSeed = seed;
		this.random = new Uniform(0, 1, randomSeed);
	}
	
	public InformationTheoreticCoClustering(double[][] dataMat, int rowClusterNum, int colClusterNum, int seed) {
		this(dataMat.length, dataMat[0].length, rowClusterNum, colClusterNum, seed);
		
		int col = dataMat[0].length;
		dtm = new ArrayList<DoubleMatrix1D>();
		for (int i = 0; i < dataMat.length; ++i) {
			if (dataMat[i].length != col) {
				System.err.println("Column dosen't match!");
			}
			dtm.add(new ColtSparseVector(dataMat[i]));
		}
	}

	public InformationTheoreticCoClustering(DoubleMatrix1D[] dataMat, int rowClusterNum, int colClusterNum, int seed) {
		this(dataMat.length, dataMat[0].size(), rowClusterNum, colClusterNum, seed);
	
		int col = dataMat[0].size();
		dtm = new ArrayList<DoubleMatrix1D>();
		for (int i = 0; i < dataMat.length; ++i) {
			if (dataMat[i].size() != col) {
				System.err.println("Column dosen't match!");
			}
			DoubleMatrix1D sample = null;
			sample = new ColtSparseVector(dataMat[i].size());
			sample.assign(dataMat[i]);
			sample.trimToSize();
			dtm.add(sample);
		}
	}
	
	public InformationTheoreticCoClustering(List<DoubleMatrix1D> dataMat, int rowClusterNum, int colClusterNum, int seed) {
		this(dataMat.size(), dataMat.get(0).size(), rowClusterNum, colClusterNum, seed);
		
		int col = dataMat.get(0).size();
		dtm = new ArrayList<DoubleMatrix1D>();
		for (int i = 0; i < dataMat.size(); ++i) {
			if (dataMat.get(i).size() != col) {
				System.err.println("Column dosen't match!");
			}
			DoubleMatrix1D sample = null;
			sample = new ColtSparseVector(dataMat.get(i).size());
			sample.assign(dataMat.get(i));
			sample.trimToSize();
			dtm.add(sample);
		}
	}
	
//	public InformationTheoreticCoClustering(DoubleMatrix2D dataMat, int rowClusterNum, int colClusterNum) {
//	this(dataMat.rows(), dataMat.columns(), rowClusterNum, colClusterNum);
//	
//	dtm = new ArrayList<DoubleMatrix1D>();
//	for (int i = 0; i < dataMat.rows(); ++i) {
//		SparseDoubleMatrix1D vector = new SparseDoubleMatrix1D(dataMat.columns());
//		vector.assign(dataMat.viewRow(i));
//		dtm.add(vector);
//	}
//}

	public void setDebug(boolean isDebug) {
		this.isDebug = isDebug;
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
				GeneralKmeans kmeanscluster = new GeneralKmeans(dtm, rcNum, "maxmin", randomSeed);
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
				
				GeneralKmeans kmeanscluster = new GeneralKmeans(dtmTrans, ccNum, "maxmin", randomSeed);
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
	
	protected void update() {
		
//		Timing time = new Timing();
		
		// check empty class for row
		int[] datanum = new int[rcNum];
		Arrays.fill(datanum, 0);
		for (int i = 0; i < memberC1[1].size(); ++i) {
			datanum[(int)memberC1[1].getQuick(i)] += 1;
		}
		int maxnum = 0;
		int minnum = Integer.MAX_VALUE;
		int maxindex = 0;
		int minindex = 0;
		for (int i = 0; i < datanum.length; ++i) {
			if (datanum[i] > maxnum) {
				maxnum = datanum[i];
				maxindex = i;
			}
			if (datanum[i] < minnum) {
				minnum = datanum[i];
				minindex = i;
			}
		}
		if (minnum == 0) {
			boolean modifyFlag = false;
			for (int i = 0; i < memberC1[1].size(); ++i) {
				if (memberC1[1].getQuick(i) == maxindex && random.nextBoolean() == true) {
					modifyFlag = true;
					memberC1[1].setQuick(i, minindex);
					break;
				}
			}
			if (modifyFlag == false) {
				for (int i = 0; i < memberC1[1].size(); ++i) {
					if (memberC1[1].getQuick(i) == maxindex) {
						memberC1[1].setQuick(i, minindex);
						break;
					}
				}				
			}
		}

		// check empty class for column
		datanum = new int[ccNum];
		Arrays.fill(datanum, 0);
		for (int i = 0; i < memberC2[1].size(); ++i) {
			datanum[(int)memberC2[1].getQuick(i)] += 1;
		}
		maxnum = 0;
		minnum = Integer.MAX_VALUE;
		maxindex = 0;
		minindex = 0;
		for (int i = 0; i < datanum.length; ++i) {
			if (datanum[i] > maxnum) {
				maxnum = datanum[i];
				maxindex = i;
			}
			if (datanum[i] < minnum) {
				minnum = datanum[i];
				minindex = i;
			}
		}
		if (minnum == 0) {
			boolean modifyFlag = false;
			for (int i = 0; i < memberC2[1].size(); ++i) {
				if (memberC2[1].getQuick(i) == maxindex && random.nextBoolean() == true) {
					modifyFlag = true;
					memberC2[1].setQuick(i, minindex);
					break;
				}
			}
			if (modifyFlag == false) {
				for (int i = 0; i < memberC2[1].size(); ++i) {
					if (memberC2[1].getQuick(i) == maxindex) {
						memberC2[1].setQuick(i, minindex);
						break;
					}
				}				
			}
		}
		
		if (isProfiling == true) {
			System.out.println("    Update check empty class: ");
		}

		// variational method q
	
		qxbyb.assign(0.0);
		for (int i = 0; i < rowNum; ++i) {
			IntArrayList indexList = new IntArrayList();
			DoubleArrayList valueList = new DoubleArrayList();
			DoubleMatrix1D vector = dtm.get(i);
			vector.getNonZeros(indexList, valueList);
			for (int j = 0; j < indexList.size(); ++j) {
				int index = indexList.get(j);
				double value = valueList.get(j);
				int indexi = (int)memberC1[1].getQuick(i);
				int indexj = (int)memberC2[1].getQuick(index);
				qxbyb.setQuick(indexi, indexj, qxbyb.getQuick(indexi,indexj) + value);
			}	
		}
		
		if (isProfiling == true) {
			System.out.println("    Update qxbyb: ");
		}
		
		qxb = Matrix2DUtil.matrixSum(qxbyb, 2);
		qyb = Matrix2DUtil.matrixSum(qxbyb, 1);
		
		if (isProfiling == true) {
			System.out.println("    Update qxb qyb: ");
		}

		// q(x| bar{x}) q(y| bar{y})
		for (int i = 0; i < qx_xb.size(); ++i) {
			qx_xb.get(i).assign(0.0);
		}
		for (int i = 0; i < rowNum; ++i) {
			int indexi = (int)memberC1[1].getQuick(i);
			qx_xb.get(i).setQuick(indexi, qx.getQuick(i) / (qxb.getQuick(indexi) + Double.MIN_VALUE));
		}
		for (int i = 0; i < qy_yb.size(); ++i) {
			qy_yb.get(i).assign(0.0);
		}
		for (int i = 0; i < columnNum; ++i) {
			int indexi = (int)memberC2[1].getQuick(i);
			qy_yb.get(i).setQuick(indexi, qy.getQuick(i) / (qyb.getQuick(indexi) + Double.MIN_VALUE));
		}
		
		if (isProfiling == true) {
			System.out.println("    Update qx_xb qy_yb: ");
		}
		
		for (int i = 0; i < rowNum; ++i) {
			IntArrayList indexList = new IntArrayList();
			DoubleArrayList valueList = new DoubleArrayList();
			DoubleMatrix1D vector = dtm.get(i);
			vector.getNonZeros(indexList, valueList);
			for (int j = 0; j < indexList.size(); ++j) {
				int index = indexList.get(j);
				int indexi = (int)memberC1[1].getQuick(i);
				int indexj = (int)memberC2[1].getQuick(index);
				double valuenew = qxbyb.getQuick(indexi, indexj)
					* qx_xb.get(i).getQuick(indexi)
					* qy_yb.get(index).getQuick(indexj);
				qxy.get(i).setQuick(index, valuenew);
			}	
		}
		if (isProfiling == true) {
			System.out.println("    Update qxy: ");
		}
		

	}
	
	public void estimate() {
//		Timing timing = new Timing();
		initialization();
		System.out.println("ITCC initialization:");
		estimate_core();
	}
	
	public void estimate_core() {
		
		if (dtm == null) {
			System.err.println("Error: no data loaded");
		}
//		Timing timing = new Timing();
	
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
			
//			for (int i = 0; i < c1.size(); ++i) {
//				for (int j = 0; j < c1.get(i).size(); ++j) {
//					double sum = 0.0;
//					for (int k = 0; k < qy_x.size(); ++k) {
//						double temp = qy_x.get(k).getQuick(i);
//						if (temp > 0.0) {
//							sum += temp
//								* ( ( Math.log( temp ) - qy_xb.get(k).getQuick(j) ) / log2 );
//						}
//					}// end k
//					c1.get(i).setQuick(j, sum);
//				}// end j
//			}// end i
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
//				for (int i = 0; i < c1.size(); ++i) {
//					double temp = qy_x.get(k).getQuick(i);
//					if (temp > 0.0) {
//						for (int j = 0; j < c1.get(i).size(); ++j) {
//							double sum = c1.get(i).getQuick(j);
//							sum += temp
//								* ( ( Math.log( temp ) - qy_xb.get(k).getQuick(j) ) / log2 );
//							c1.get(i).setQuick(j, sum);
//						}
//					}
//				}
			} 
			if (isProfiling == true) {
				System.out.println("    Estimate c1: ");
			}
			// update cluster centers and variational parameters
			memberC1 = Matrix2DUtil.matrixMin(c1, 2);
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
//			for (int i = 0; i < c2.size(); ++i) {
//				for (int j = 0; j < c2.get(i).size(); ++j) {
//					double sum = 0.0;
//					for (int k = 0; k < qx_y.size(); ++k) {
//						double temp = qx_y.get(k).getQuick(i);
//						if (temp > 0.0) {
//							sum += temp
//								* ( ( Math.log( temp ) - qx_yb.get(k).getQuick(j) ) / log2 );
//
//						}
//					}// end k
//					c2.get(i).setQuick(j, sum);
//				}// end j
//			}// end i	
			
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
//				for (int i = 0; i < c2.size(); ++i) {
//					double temp = qx_y.get(k).getQuick(i);
//					if (temp > 0.0) {
//						for (int j = 0; j < c2.get(i).size(); ++j) {
//							double sum = c2.get(i).getQuick(j);
//							sum += temp
//								* ( ( Math.log( temp ) - qx_yb.get(k).getQuick(j) ) / log2 );
//							c2.get(i).setQuick(j, sum);
//						}
//					}
//				}
			} 
			if (isProfiling == true) {
				System.out.println("    Estimate c2: ");
			}
			
			// update cluster centers and variational parameters
			memberC2 = Matrix2DUtil.matrixMin(c2, 2);
			if (isProfiling == true) {
				System.out.println("    Estimate memberC2: ");
			}
			
			update();
			
			fullGC();
			
			// find total cost
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
			
			System.out.println(" Interaion " + iter + " deltaMeans = " + delta + ": ");
			
			if (delta < MIN_DELTA) {
//				if (isDebug == true) 
				{
					System.out.println("Infomation Theoretial Co-clustering. Finished!");
				}
				break;
			}
			oldCost = cost;
		}// end iteration
			
	}
	
	protected void fullGC() {
		Runtime rt = Runtime.getRuntime();
		long isFree = rt.freeMemory();
		long wasFree;
		do {
			wasFree = isFree;
			rt.runFinalization();
			rt.gc();
			isFree = rt.freeMemory();
		} while (isFree > wasFree);
	}
	
	public void setRandomSeed(int seed) {
		randomSeed = seed;
		random = new Uniform(0, 1, randomSeed);
	}
	
	public void setMaxIterNum(int num) {
		MAX_ITER = num;
	}
	
	public void setMinDelta(double delta) {
		MIN_DELTA = delta;
	}

	public void setTolerance(double tol) {
		TOLERANCE = tol;
	}
	
	public void setRowClusterLabels(int[] labels) {
		double[] doubleLabels = new double[labels.length];
		for (int i = 0; i < labels.length;  ++i) {
			doubleLabels[i] = labels[i];
		}
		if (memberC1 != null && memberC1[1].size() == labels.length) {
			memberC1[1].assign(doubleLabels);
		} else if (memberC1 == null) {
			memberC1 = new ColtSparseVector[2];
			memberC1[0] = new ColtSparseVector(labels.length);
			memberC1[1] = new ColtSparseVector(labels.length);
			memberC1[1].assign(doubleLabels);
		}
	}

	public void setColumnClusterLabels(int[] labels) {
		double[] doubleLabels = new double[labels.length];
		for (int i = 0; i < labels.length;  ++i) {
			doubleLabels[i] = labels[i];
		}
		if (memberC2 != null && memberC2[1].size() == labels.length) {
			memberC2[1].assign(doubleLabels);
		} else if (memberC2 == null) {
			memberC2 = new ColtSparseVector[2];
			memberC2[0] = new ColtSparseVector(labels.length);
			memberC2[1] = new ColtSparseVector(labels.length);
			memberC2[1].assign(doubleLabels);
		}
	}
	
	public int[] getRowClusterLabels() {
		if (memberC1 != null) {
			double[] doubleLabels = memberC1[1].toArray();
			int[] labels = new int[doubleLabels.length];
			for (int i = 0; i < doubleLabels.length;  ++i) {
				labels[i] = (int)doubleLabels[i];
			}
			return labels;
		} else {
			System.err.println("Haven't trained...");
			return null;
		}
	}

	public int[] getColumnClusterLabels() {
		if (memberC2 != null) {
			double[] doubleLabels = memberC2[1].toArray();
			int[] labels = new int[doubleLabels.length];
			for (int i = 0; i < doubleLabels.length;  ++i) {
				labels[i] = (int)doubleLabels[i];
			}
			return labels;
		} else {
			System.err.println("Haven't trained...");
			return null;
		}
	}

	public  DoubleMatrix2D getQxbyb() {
		if (qxbyb != null) {
			return qxbyb;
		} else {
			System.err.println("Haven't trained...");
			return null;
		}
	}
	
	public List<DoubleMatrix1D> getQx_xb() {
		if (qx_xb != null) {
			return qx_xb;
		} else {
			System.err.println("Haven't trained...");
			return null;
		}
	}
	
	public List<DoubleMatrix1D> getQy_yb() {
		if (qy_yb != null) {
			return qy_yb;
		} else {
			System.err.println("Haven't trained...");
			return null;
		}
	}
	
	public List<DoubleMatrix1D> getRowCentersList() {
		List<DoubleMatrix1D> centers = new ArrayList<DoubleMatrix1D>();
		for (int i = 0; i < this.rcNum; ++i) {
			DoubleMatrix1D center = new ColtSparseVector(this.columnNum);
			for (int n = 0; n < this.columnNum; ++n) {
				double value = 0.0;
				for (int j = 0; j < this.ccNum; ++j) {
					value += this.qxbyb.get(i, j) * this.qy_yb.get(n).get(j);
				}
				if (value != 0) {
					center.set(n, value);
				}
			}
			centers.add(center);
		}
		return centers;
	}
	
	// compared to getRowCentersList, this function return all the word clusters.
	public List<DoubleMatrix1D> getWordCentersList() {
		List<DoubleMatrix1D> centers = new ArrayList<DoubleMatrix1D>();
		for (int i = 0; i < this.ccNum; ++i) {
			DoubleMatrix1D center = new ColtSparseVector(this.columnNum);
			for (int n = 0; n < this.columnNum; ++n) {
				double value = this.qy_yb.get(n).get(i);
				center.set(n, value);
			}
			centers.add(center);
		}
		return centers;
	}
	
	public int getNumInstances() {
		return this.dtm.size();
	}
	
	
	// mini test
	public static void main (String[] args) {

//		double[][] feature = {
//				  {0.05, 0.05, 0.05, 0, 0, 0},
//				  {0.05, 0.05, 0.05, 0, 0, 0},
//				  {0, 0, 0, 0.05, 0.05, 0.05},
//				  {0, 0, 0, 0.05, 0.05, 0.05},
//				  {0.04, 0.04, 0, 0.04, 0.04, 0.04},
//				  {0.04, 0.04, 0.04, 0, 0.04, 0.04},
//				  };	
//		double[][] feature = {
//				  {0.03, 0.05, 0.05, 0, 0, 0},
//				  {0.05, 0.03, 0.05, 0, 0, 0},
//				  {0.03, 0.03, 0.03, 0.05, 0.04, 0.05},
//				  {0, 0, 0, 0.05, 0.05, 0.04},
//				  {0.04, 0.04, 0.03, 0.03, 0.04, 0.04},
//				  {0.03, 0.04, 0.03, 0.03, 0.04, 0.03},
//				  };	
		
		double[][] feature = {
				  {0.03, 0.05, 0.05, 0,    0,    0},
				  {0.05, 0.03, 0.05, 0,    0,    0},
				  {0.01, 0.01, 0.01, 0.05, 0.04, 0.05},
				  {0.04, 0.04, 0.04, 0.05, 0.05, 0.04},
				  {0.04, 0.04, 0.03, 0.03, 0.04, 0.04},
				  {0.03, 0.04, 0.03, 0.03, 0.04, 0.03},
				  };	
		InformationTheoreticCoClustering itcc = new InformationTheoreticCoClustering(feature, 3, 2);
		itcc.setRandomSeed(11122345);
//		double[] rlabels = {0, 0, 1, 1, 2, 2};
//		itcc.setRowClusterLabels(rlabels);
//		double[] clabels = {0, 0, 0, 1, 1, 1};
//		itcc.setColumnClusterLabels(clabels);
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
		
		
		
		
	}
}
