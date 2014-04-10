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
import cern.jet.random.Uniform;



/*
 * @inproceedings{DBLP:conf/sdm/WangLZ08,
  author    = {Fei Wang and
               Tao Li and
               Changshui Zhang},
  title     = {Semi-Supervised Clustering via Matrix Factorization},
  booktitle = {SDM},
  year      = {2008},
  pages     = {1-12},
  ee        = {http://www.siam.org/proceedings/datamining/2008/dm08_01_Wang.pdf},
  crossref  = {DBLP:conf/sdm/2008},
  bibsource = {DBLP, http://dblp.uni-trier.de}
}

Convex and Semi-Nonnegative Matrix Factorizations Export 
by: Chris Ding, Tao Li, Michael I. Jordan
IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 32, No. 1. (21 November 2010), pp. 45-55.

 */

public class ConstraintSemiNMF {
	protected int MAX_ITER = 5;
	protected double MIN_DELTA = 0.01;
	protected double weight = 0.5;

	protected double TOLERANCE = 0.00000000000000000001;
	
	protected boolean isDebug = true;
	
	protected boolean isProfiling = false;

	protected boolean isInitUseKmeans = true;

	// matrix row and column number
	protected int rowNum;
	// cluster number
	protected int rcNum;
	
	// data term matrix
	protected List<DoubleMatrix1D> dtm = null;

	protected List<DoubleMatrix1D> c1 = null;

	protected DoubleMatrix1D[] memberC1 = null;
	
	private List<DoubleMatrix1D> rCMat;

	private List<DoubleMatrix1D> rCMatPos;
	private List<DoubleMatrix1D> rCMatNeg;
	
	private int[][] rowMustlink = null;
	private int[][] rowCannotlink = null; 
	
	private double[] rowMustWeight = null;
	private double[] rowCannotWeight = null;
	
	protected int randomSeed = 0;
	protected Uniform random = null;
	
	
	public ConstraintSemiNMF(double[][] dataMat, int rowClusterNum) {
		this(dataMat, rowClusterNum, 0);
	}
	public ConstraintSemiNMF(DoubleMatrix1D[] dataMat, int rowClusterNum) {
		this(dataMat, rowClusterNum, 0);
	}
	public ConstraintSemiNMF(List<DoubleMatrix1D> dataMat, int rowClusterNum) {
		this(dataMat, rowClusterNum, 0);
	}

	public ConstraintSemiNMF(int rowNum, int rowClusterNum, int seed) {
		this.rcNum = rowClusterNum;
		this.rowNum = rowNum;
		this.randomSeed = seed;
		this.random = new Uniform(0, 1, randomSeed);
	}
	
	public ConstraintSemiNMF(double[][] dataMat, int rowClusterNum, int seed) {
		this(dataMat.length, rowClusterNum, seed);
		
		int col = dataMat[0].length;
		dtm = new ArrayList<DoubleMatrix1D>();
		for (int i = 0; i < dataMat.length; ++i) {
			if (dataMat[i].length != col) {
				System.err.println("Column dosen't match!");
			}
			dtm.add(new ColtSparseVector(dataMat[i]));
		}
	}

	public ConstraintSemiNMF(DoubleMatrix1D[] dataMat, int rowClusterNum, int seed) {
		this(dataMat.length, rowClusterNum, seed);
	
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
	
	public ConstraintSemiNMF(List<DoubleMatrix1D> dataMat, int rowClusterNum, int seed) {
		this(dataMat.size(), rowClusterNum, seed);
		
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
	
	public ConstraintSemiNMF(double[][] dataMat, 
			int rowClusterNum, 
			int[][] rowMustlink, int[][] rowCannotlink) {
		this(dataMat, rowClusterNum);
		this.rowMustlink = rowMustlink;
		this.rowCannotlink = rowCannotlink;

	}

	public ConstraintSemiNMF(DoubleMatrix1D[] dataMat, 
			int rowClusterNum,
			int[][] rowMustlink, int[][] rowCannotlink) {
		this(dataMat, rowClusterNum);
		this.rowMustlink = rowMustlink;
		this.rowCannotlink = rowCannotlink;
	}

	public ConstraintSemiNMF(List<DoubleMatrix1D> dataMat, 
			int rowClusterNum, 
			int[][] rowMustlink, int[][] rowCannotlink) {
		this(dataMat, rowClusterNum);
		this.rowMustlink = rowMustlink;
		this.rowCannotlink = rowCannotlink;
	}

	public ConstraintSemiNMF(double[][] dataMat, 
													int rowClusterNum,
													int[][] rowMustlink, int[][] rowCannotlink,
													int seed) {
		this(dataMat, rowClusterNum, seed);
		this.rowMustlink = rowMustlink;
		this.rowCannotlink = rowCannotlink;
	}
	
	public ConstraintSemiNMF(DoubleMatrix1D[] dataMat, 
													int rowClusterNum, int colClusterNum, 
													int[][] rowMustlink, int[][] rowCannotlink,
													int seed) {
		this(dataMat, rowClusterNum, seed);
		this.rowMustlink = rowMustlink;
		this.rowCannotlink = rowCannotlink;
	}
	
	public ConstraintSemiNMF(List<DoubleMatrix1D> dataMat, 
													int rowClusterNum, 
													int[][] rowMustlink, int[][] rowCannotlink,
													int seed) {
		this(dataMat, rowClusterNum, seed);
		this.rowMustlink = rowMustlink;
		this.rowCannotlink = rowCannotlink;
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

	protected void initialization() {
		
//		Timing time = new Timing();
		if (dtm == null) {
			System.err.println("Error: no data loaded!");
			return;
		}
		if (isDebug == true) {
			System.out.println("ConstraintSemiNMF Initializaiton... ");
		}
		
		// initialize cluster membership matrix
		if (isDebug == true) {
			System.out.println("Initialize c1 c2...");
		}
		
//		DoubleMatrix1D mean = mean(dtm);
		
		// data normalization
		for (int i = 0; i < dtm.size(); ++i) {
			double sum = 0.0;
			IntArrayList indexList = new IntArrayList();
			DoubleArrayList valueList = new DoubleArrayList();
			dtm.get(i).getNonZeros(indexList, valueList);
			for (int j = 0; j < indexList.size(); ++j) {
				sum += valueList.get(j) * valueList.get(j);
			}
			sum = Math.sqrt(sum);
			for (int j = 0; j < indexList.size(); ++j) {
				int index = indexList.get(j);
				double value = valueList.get(j);
				dtm.get(i).set(index, value/(sum+Double.MIN_NORMAL));
//				DoubleMatrix1D vector = dtm.get(i);
//				vector = null;
			}
		}
		
		c1 = new ArrayList<DoubleMatrix1D>();
		for (int i = 0; i < rowNum; ++i){
			c1.add(new ColtDenseVector(rcNum));
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
					c1.get(i).setQuick(j, 1.2);
				} else {
					c1.get(i).setQuick(j, 0.2);
				}
			}			
		}			
		if (isProfiling == true) {
			System.out.println("    Initialize c1: ");
		}
		
	}
	
	private void initializeConstriants() {
		
		// for row pairs
		rCMat = new ArrayList<DoubleMatrix1D>();
		for (int i = 0; i < rowNum; ++i) {
			DoubleMatrix1D vector = new ColtSparseVector(rowNum);
			rCMat.add(vector);
		}
		rCMatPos = new ArrayList<DoubleMatrix1D>();
		for (int i = 0; i < rowNum; ++i) {
			DoubleMatrix1D vector = new ColtSparseVector(rowNum);
			rCMatPos.add(vector);
		}
		rCMatNeg = new ArrayList<DoubleMatrix1D>();
		for (int i = 0; i < rowNum; ++i) {
			DoubleMatrix1D vector = new ColtSparseVector(rowNum);
			rCMatNeg.add(vector);
		}
		if (rowMustlink != null) {
			if (this.rowMustWeight == null) {
				this.rowMustWeight = new double[rowMustlink.length];
				for (int i = 0; i < rowMustWeight.length; ++i) {
					this.rowMustWeight[i] = weight;//(Math.sqrt(this.rowNum) + Double.MIN_NORMAL);
				}
			} 

			for (int i = 0; i < rowMustlink.length; ++i) {
				int from = rowMustlink[i][0];
				int to = rowMustlink[i][1];
				rCMatNeg.get(from).set(to, rowMustWeight[i]);
				rCMat.get(from).set(to, 0 - rowMustWeight[i]);
				rCMatNeg.get(to).set(from, rowMustWeight[i]);
				rCMat.get(to).set(from, 0 - rowMustWeight[i]);
			}
		}
		if (rowCannotlink != null) {
			if (this.rowCannotWeight == null) {
				this.rowCannotWeight = new double[rowCannotlink.length];
				for (int i = 0; i < rowCannotWeight.length; ++i) {
					this.rowCannotWeight[i] = weight;//(Math.sqrt(this.rowNum) + Double.MIN_NORMAL);
				}
			} 

			for (int i = 0; i < rowCannotlink.length; ++i) {
				int from = rowCannotlink[i][0];
				int to = rowCannotlink[i][1];
				rCMatPos.get(from).set(to, rowCannotWeight[i]);
				rCMat.get(from).set(to, rowCannotWeight[i]);
				rCMatPos.get(to).set(from, rowCannotWeight[i]);
				rCMat.get(to).set(from, rowCannotWeight[i]);
			}
		}
		
		
	}
	
	public void estimate() {
		
		if (dtm == null) {
			System.err.println("Error: no data loaded");
		}
//		Timing timing = new Timing();
	
		initialization();

		System.out.println("ConstraintSemiNMF initialization:");
		
		initializeConstriants();

		System.out.println("ConstraintSemiNMF initialization of constraints:");
		
//		Timing time = new Timing();
		double oldCost = Double.MAX_VALUE;
		double cost = 0.0;
		double delta = 0.0;
		

		List<DoubleMatrix1D> F = null;
		List<DoubleMatrix1D> FF = null;
		List<DoubleMatrix1D> XF = null;
		
		List<DoubleMatrix1D> FFPos = null;
		List<DoubleMatrix1D> FFNeg = null;
		
		List<DoubleMatrix1D> temp1 = null;
		List<DoubleMatrix1D> temp2 = null;
		List<DoubleMatrix1D> temp3 = null;
		List<DoubleMatrix1D> temp4 = null;
		List<DoubleMatrix1D> temp5 = null;
		List<DoubleMatrix1D> temp6 = null;
		
		for (int iter = 0; iter < MAX_ITER; ++iter) {
			
			// compute F:
			temp1 = Matrix2DUtil.SparseTransposeMultSparse(c1, c1);
			temp2 = Matrix2DUtil.inverseSparse(temp1);
			temp1 = null;
			temp3 = Matrix2DUtil.SparseMultSparse(c1, temp2);
			temp2 = null;
			F = Matrix2DUtil.SparseTransposeMultSparse(dtm, temp3);
			temp3 = null;
			
			///////////////////////////////////
			// for row
			///////////////////////////////////	
			
			// compute FF:
			
			FF = Matrix2DUtil.SparseTransposeMultSparse(F, F);
			FFPos = new ArrayList<DoubleMatrix1D>();
			FFNeg = new ArrayList<DoubleMatrix1D>();
			for (int i = 0; i < FF.size(); ++i) {
				FFPos.add(new ColtSparseVector(FF.get(i).size()));
				FFNeg.add(new ColtSparseVector(FF.get(i).size()));
			}
			for (int i = 0; i < FF.size(); ++i) {
				IntArrayList indexList = new IntArrayList();
				DoubleArrayList valueList = new DoubleArrayList();
				FF.get(i).getNonZeros(indexList, valueList);
				for (int j = 0; j < indexList.size(); ++j) {
					int index = indexList.get(j);
					double value = valueList.get(j);
					if (value > 0) {
						FFPos.get(i).set(index, value);
					} else if (value < 0) {
						FFNeg.get(i).set(index, -value);
					}
				}
			}
			
			// compute XF:
			XF = Matrix2DUtil.SparseMultSparse(dtm, F);
			
			temp3 = Matrix2DUtil.SparseMultSparse(c1, FFNeg);
			FFNeg = null;
			temp4 = Matrix2DUtil.SparseMultSparse(c1, FFPos);
			FFPos = null;
			
			temp5 = Matrix2DUtil.SparseMultSparse(rCMatNeg, c1);
			temp6 = Matrix2DUtil.SparseMultSparse(rCMatPos, c1);
			
			double t0, t1, t2, t3, t4, t5, t6;
			for (int i = 0; i < c1.size(); ++i) {
				for (int j = 0; j < c1.get(i).size(); ++j) {
					t1 = XF.get(i).get(j);
					if (t1 < 0) {
						t1 = 0;
					}
					t2 = temp3.get(i).get(j);
					t3 = temp5.get(i).get(j);
					t4 = XF.get(i).get(j);
					if (t4 > 0) {
						t4 = 0;
					} else {
						t4 = -t4;
					}
					t5 = temp4.get(i).get(j);
					t6 = temp6.get(i).get(j);
					
					t0 = c1.get(i).get(j);
					
					t0 = t0 * Math.sqrt((t1 + t2 + t3) / (t4 + t5 + t6 + this.TOLERANCE));
					
//					t0 = t0 * Math.sqrt( (t1 + t2 + 1*Math.exp (0 - 1/t3)) / 
//							(t4 + t5 + 1*Math.exp (0 - 1/t6) + this.TOLERANCE) );
//					
//					if (t0 > 1) {
//						t0 = 1;
//					}
					
					c1.get(i).set(j, t0);
				}
			}
			XF = null;
			temp3 = null;
			temp4 = null;
			temp5 = null;
			temp6 = null;
			
			fullGC();
			
			// Compute cost
			cost = 0.0;
//			temp1 = Matrix2DUtil.SparseMultSparseTranspose(c1, F);
//			for (int i = 0; i < dtm.size(); ++i) {
//				for (int j = 0; j < dtm.get(i).size(); ++j) {
//					cost += Math.pow((dtm.get(i).get(j) - temp1.get(i).get(j)), 2);
//				}
//			}
//			temp1 = null;
			for (int i = 0; i < dtm.size(); ++i) {
				for (int j = 0; j < dtm.get(i).size(); ++j) {
					double value = Matrix2DUtil.product(c1.get(i), F.get(j));
					cost += Math.pow((dtm.get(i).get(j) - value), 2);
				}
			}
			
			temp1 = Matrix2DUtil.SparseMultSparse(rCMat, c1);
//			temp2 = Matrix2DUtil.SparseTransposeMultSparse(c1, temp1);
//			temp1 = null;
//			for (int i = 0; i < temp2.size(); ++i) {
//				cost += Math.pow(temp2.get(i).get(i), 2);
//			}
//			temp2 = null;
			List<DoubleMatrix1D> c1T = Matrix2DUtil.getSparseTranspose(c1);
			List<DoubleMatrix1D> temp1T = Matrix2DUtil.getSparseTranspose(temp1);
			for (int i = 0; i < c1T.size(); ++i) {
				double value = Matrix2DUtil.product(c1T.get(i), temp1T.get(i));
				cost += Math.pow(value, 1);
			}
			c1T = null;
			temp1T = null;
			temp1 = null;
			
			cost = Math.abs(cost);
			
			delta = Math.abs(oldCost - cost) / (oldCost + this.TOLERANCE);
			
			memberC1 = Matrix2DUtil.matrixMax(c1, 2);		
			
			System.out.println(" Constraint Interaion " + iter + " cost = " + cost + " deltaMeans = " + delta + ": ");
			
			if (delta < MIN_DELTA) {
//				if (isDebug == true) 
				{
					System.out.println("Constraint SemiNMF. Finished!");
				}
				break;
			}
			oldCost = cost;
		}// end iteration
		
	}
	
	private void fullGC() {
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
	
	public int getNumInstances() {
		return this.dtm.size();
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

	// mini test
	public static void main (String[] args) {

		double[][] feature = {
				  {0.05, 0.05, 0.05, 0, 0, 0},
				  {0.05, 0.05, 0.05, 0, 0, 0},
				  {0, 0, 0, 0.05, 0.05, 0.05},
				  {0, 0, 0, 0.05, 0.05, 0.05},
				  {0.04, 0.04, 0, 0.04, 0.04, 0.04},
				  {0.04, 0.04, 0.04, 0, 0.04, 0.04},
				  };	
//		double[][] feature = {
//				  {0.03, 0.05, 0.05, 0, 0, 0},
//				  {0.05, 0.03, 0.05, 0, 0, 0},
//				  {0.03, 0.03, 0.03, 0.05, 0.04, 0.05},
//				  {0, 0, 0, 0.05, 0.05, 0.04},
//				  {0.04, 0.04, 0.03, 0.03, 0.04, 0.04},
//				  {0.03, 0.04, 0.03, 0.03, 0.04, 0.03},
//				  };	
		ConstraintSemiNMF sstnmf = new ConstraintSemiNMF(feature, 3);
		sstnmf.setRandomSeed(1);
		int[] rlabels = {0, 0, 1, 1, 2, 2};
		sstnmf.setRowClusterLabels(rlabels);
		sstnmf.estimate();
		int[] rlabel = sstnmf.getRowClusterLabels();
		for (int i = 0; i < rlabel.length; ++i) {
			System.out.print(rlabel[i] + " ");
		}
		System.out.println("");
	}
}
