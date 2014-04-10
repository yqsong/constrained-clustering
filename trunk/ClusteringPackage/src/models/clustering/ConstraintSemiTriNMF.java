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

public class ConstraintSemiTriNMF {
	protected int MAX_ITER = 5;
	protected double MIN_DELTA = 0.01;
	protected double weight = 0.5;

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
	
	private List<DoubleMatrix1D> rCMat;
	private List<DoubleMatrix1D> cCMat;
	
	private List<DoubleMatrix1D> rCMatPos;
	private List<DoubleMatrix1D> cCMatPos;
	private List<DoubleMatrix1D> rCMatNeg;
	private List<DoubleMatrix1D> cCMatNeg;
	
	
	private int[][] rowMustlink = null;
	private int[][] rowCannotlink = null; 
	private int[][] columnMustlink = null;
	private int[][] columnCannotlink = null;
	
	private double[] rowMustWeight = null;
	private double[] rowCannotWeight = null;
	private double[] columnMustWeight = null;
	private double[] columnCannotWeight = null;
	
	protected int randomSeed = 0;
	protected Uniform random = null;
	
	
	public ConstraintSemiTriNMF(double[][] dataMat, int rowClusterNum, int colClusterNum) {
		this(dataMat, rowClusterNum, colClusterNum, 0);
	}
	public ConstraintSemiTriNMF(DoubleMatrix1D[] dataMat, int rowClusterNum, int colClusterNum) {
		this(dataMat, rowClusterNum, colClusterNum, 0);
	}
	public ConstraintSemiTriNMF(List<DoubleMatrix1D> dataMat, int rowClusterNum, int colClusterNum) {
		this(dataMat, rowClusterNum, colClusterNum, 0);
	}

	public ConstraintSemiTriNMF(int rowNum, int columnNum, int rowClusterNum, int colClusterNum, int seed) {
		this.rcNum = rowClusterNum;
		this.ccNum = colClusterNum;
		this.rowNum = rowNum;
		this.columnNum = columnNum;
		this.randomSeed = seed;
		this.random = new Uniform(0, 1, randomSeed);
	}
	
	public ConstraintSemiTriNMF(double[][] dataMat, int rowClusterNum, int colClusterNum, int seed) {
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

	public ConstraintSemiTriNMF(DoubleMatrix1D[] dataMat, int rowClusterNum, int colClusterNum, int seed) {
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
	
	public ConstraintSemiTriNMF(List<DoubleMatrix1D> dataMat, int rowClusterNum, int colClusterNum, int seed) {
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
	
	public ConstraintSemiTriNMF(double[][] dataMat, 
			int rowClusterNum, int colClusterNum, 
			int[][] rowMustlink, int[][] rowCannotlink, 
			int[][] columnMustlink, int[][] columnCannotlink) {
		this(dataMat, rowClusterNum, colClusterNum);
		this.rowMustlink = rowMustlink;
		this.rowCannotlink = rowCannotlink;
		this.columnMustlink = columnMustlink;
		this.columnCannotlink = columnCannotlink;

	}

	public ConstraintSemiTriNMF(DoubleMatrix1D[] dataMat, 
			int rowClusterNum, int colClusterNum, 
			int[][] rowMustlink, int[][] rowCannotlink, 
			int[][] columnMustlink, int[][] columnCannotlink) {
		this(dataMat, rowClusterNum, colClusterNum);
		this.rowMustlink = rowMustlink;
		this.rowCannotlink = rowCannotlink;
		this.columnMustlink = columnMustlink;
		this.columnCannotlink = columnCannotlink;
	}

	public ConstraintSemiTriNMF(List<DoubleMatrix1D> dataMat, 
			int rowClusterNum, int colClusterNum, 
			int[][] rowMustlink, int[][] rowCannotlink, 
			int[][] columnMustlink, int[][] columnCannotlink) {
		this(dataMat, rowClusterNum, colClusterNum);
		this.rowMustlink = rowMustlink;
		this.rowCannotlink = rowCannotlink;
		this.columnMustlink = columnMustlink;
		this.columnCannotlink = columnCannotlink;
	}

	public ConstraintSemiTriNMF(double[][] dataMat, 
													int rowClusterNum, int colClusterNum, 
													int[][] rowMustlink, int[][] rowCannotlink, 
													int[][] columnMustlink, int[][] columnCannotlink,
													int seed) {
		this(dataMat, rowClusterNum, colClusterNum, seed);
		this.rowMustlink = rowMustlink;
		this.rowCannotlink = rowCannotlink;
		this.columnMustlink = columnMustlink;
		this.columnCannotlink = columnCannotlink;
	}
	
	public ConstraintSemiTriNMF(DoubleMatrix1D[] dataMat, 
													int rowClusterNum, int colClusterNum, 
													int[][] rowMustlink, int[][] rowCannotlink, 
													int[][] columnMustlink, int[][] columnCannotlink,
													int seed) {
		this(dataMat, rowClusterNum, colClusterNum, seed);
		this.rowMustlink = rowMustlink;
		this.rowCannotlink = rowCannotlink;
		this.columnMustlink = columnMustlink;
		this.columnCannotlink = columnCannotlink;
	}
	
	public ConstraintSemiTriNMF(List<DoubleMatrix1D> dataMat, 
													int rowClusterNum, int colClusterNum, 
													int[][] rowMustlink, int[][] rowCannotlink, 
													int[][] columnMustlink, int[][] columnCannotlink,
													int seed) {
		this(dataMat, rowClusterNum, colClusterNum, seed);
		this.rowMustlink = rowMustlink;
		this.rowCannotlink = rowCannotlink;
		this.columnMustlink = columnMustlink;
		this.columnCannotlink = columnCannotlink;
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
	
	protected void initialization() {
		
//		Timing time = new Timing();
		if (dtm == null) {
			System.err.println("Error: no data loaded!");
			return;
		}
		if (isDebug == true) {
			System.out.println("ConstraintSemiTriNMF Initializaiton... ");
		}
		
		// initialize cluster membership matrix
		if (isDebug == true) {
			System.out.println("Initialize c1 c2...");
		}
		
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
				dtm.get(i).set(index, value/sum);
			}
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
					c1.get(i).setQuick(j, 1.2);
				} else {
					c1.get(i).setQuick(j, 0.2);
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
					c2.get(i).setQuick(j, 1.2);
				} else {
					c2.get(i).setQuick(j, 0.2);
				}
			}			
		}	
		
		if (isProfiling == true) {
			System.out.println("    Initialize c2: ");
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
		
		// for column pairs
		cCMat = new ArrayList<DoubleMatrix1D>();
		for (int i = 0; i < columnNum; ++i) {
			DoubleMatrix1D vector = new ColtSparseVector(columnNum);
			cCMat.add(vector);
		}
		cCMatPos = new ArrayList<DoubleMatrix1D>();
		for (int i = 0; i < columnNum; ++i) {
			DoubleMatrix1D vector = new ColtSparseVector(columnNum);
			cCMatPos.add(vector);
		}
		cCMatNeg = new ArrayList<DoubleMatrix1D>();
		for (int i = 0; i < columnNum; ++i) {
			DoubleMatrix1D vector = new ColtSparseVector(columnNum);
			cCMatNeg.add(vector);
		}
		if (columnMustlink != null) {
			if (this.columnMustWeight == null) {
				this.columnMustWeight = new double[columnMustlink.length];
				for (int i = 0; i < columnMustWeight.length; ++i) {
					this.columnMustWeight[i] = weight;//(Math.sqrt(this.columnNum) + Double.MIN_NORMAL);
				}
			} 

			for (int i = 0; i < columnMustlink.length; ++i) {
				int from = columnMustlink[i][0];
				int to = columnMustlink[i][1];
				cCMatNeg.get(from).set(to, columnMustWeight[i]);
				cCMat.get(from).set(to, 0 - columnMustWeight[i]);
				cCMatNeg.get(to).set(from, columnMustWeight[i]);
				cCMat.get(to).set(from, 0 - columnMustWeight[i]);
			}
		}
		if (columnCannotlink != null) {
			if (this.columnCannotWeight == null) {
				this.columnCannotWeight = new double[columnCannotlink.length];
				for (int i = 0; i < columnCannotWeight.length; ++i) {
					this.columnCannotWeight[i] = weight;//(Math.sqrt(this.columnNum) + Double.MIN_NORMAL);
				}
			} 

			for (int i = 0; i < columnCannotlink.length; ++i) {
				int from = columnCannotlink[i][0];
				int to = columnCannotlink[i][1];
				cCMatPos.get(from).set(to, columnCannotWeight[i]);
				cCMat.get(from).set(to, columnCannotWeight[i]);
				cCMatPos.get(to).set(from, columnCannotWeight[i]);
				cCMat.get(to).set(from, columnCannotWeight[i]);
			}
		}
	}
	
	public void estimate() {
		
		if (dtm == null) {
			System.err.println("Error: no data loaded");
		}
//		Timing timing = new Timing();
	
		initialization();

		System.out.println("ConstraintSemiTriNMF initialization:");
		
		initializeConstriants();

		System.out.println("ConstraintSemiTriNMF initialization of constraints:");
		
//		Timing time = new Timing();
		double oldCost = Double.MAX_VALUE;
		double cost = 0.0;
		double delta = 0.0;
		
		List<DoubleMatrix1D> S = null;
		
		List<DoubleMatrix1D> A1 = null;
		List<DoubleMatrix1D> A2 = null;
		
		List<DoubleMatrix1D> A2Pos = null;
		List<DoubleMatrix1D> A2Neg = null;
		
		List<DoubleMatrix1D> temp1 = null;
		List<DoubleMatrix1D> temp2 = null;
		List<DoubleMatrix1D> temp3 = null;
		List<DoubleMatrix1D> temp4 = null;
		List<DoubleMatrix1D> temp5 = null;
		List<DoubleMatrix1D> temp6 = null;
		List<DoubleMatrix1D> temp7 = null;
		for (int iter = 0; iter < MAX_ITER; ++iter) {
			
			
			
			// compute S:
			temp1 = Matrix2DUtil.SparseTransposeMultSparse(c1, c1);
			temp2 = Matrix2DUtil.SparseTransposeMultSparse(c2, c2);
			temp3 = Matrix2DUtil.inverseSparse(temp1);
			temp4 = Matrix2DUtil.inverseSparse(temp2);
			
			temp5 = Matrix2DUtil.SparseMultSparse(c2, temp4);
			temp4 = null;
			temp6 = Matrix2DUtil.SparseMultSparse(dtm, temp5);
			temp5 = null;
			temp7 = Matrix2DUtil.SparseTransposeMultSparse(c1, temp6);
			temp6 = null;
			S = Matrix2DUtil.SparseMultSparse(temp3, temp7);
			temp3 = null;
			temp7 = null;
			
			///////////////////////////////////
			// for row
			///////////////////////////////////	
			
			// compute A1:
			temp3 = Matrix2DUtil.SparseMultSparseTranspose(c2, S);
			A1 = Matrix2DUtil.SparseMultSparse(dtm, temp3);
			temp3 = null;
			
			// compute A2:
			temp3 = Matrix2DUtil.SparseMultSparseTranspose(temp2, S);
			A2 = Matrix2DUtil.SparseMultSparse(S, temp3);
			temp3 = null;
			
			A2Pos = new ArrayList<DoubleMatrix1D>();
			A2Neg = new ArrayList<DoubleMatrix1D>();
			for (int i = 0; i < A2.size(); ++i) {
				A2Pos.add(new ColtSparseVector(A2.get(i).size()));
				A2Neg.add(new ColtSparseVector(A2.get(i).size()));
			}
			for (int i = 0; i < A2.size(); ++i) {
				IntArrayList indexList = new IntArrayList();
				DoubleArrayList valueList = new DoubleArrayList();
				A2.get(i).getNonZeros(indexList, valueList);
				for (int j = 0; j < indexList.size(); ++j) {
					int index = indexList.get(j);
					double value = valueList.get(j);
					if (value > 0) {
						A2Pos.get(i).set(index, value);
					} else if (value < 0) {
						A2Neg.get(i).set(index, -value);
					}
				}
			}
			
			A2 = null;
			
			temp3 = Matrix2DUtil.SparseMultSparse(c1, A2Neg);
			A2Neg = null;
			temp4 = Matrix2DUtil.SparseMultSparse(c1, A2Pos);
			A2Pos = null;
			
			temp5 = Matrix2DUtil.SparseMultSparse(rCMatNeg, c1);
			temp6 = Matrix2DUtil.SparseMultSparse(rCMatPos, c1);
			
			double t0, t1, t2, t3, t4, t5, t6;
			for (int i = 0; i < c1.size(); ++i) {
				for (int j = 0; j < c1.get(i).size(); ++j) {
					t1 = A1.get(i).get(j);
					if (t1 < 0) {
						t1 = 0;
					}
					t2 = temp3.get(i).get(j);
					t3 = temp5.get(i).get(j);
					t4 = A1.get(i).get(j);
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
			A1 = null;
			temp3 = null;
			temp4 = null;
			temp5 = null;
			temp6 = null;
			
			
			///////////////////////////////////
			// for column
			///////////////////////////////////	
			temp1 = Matrix2DUtil.SparseTransposeMultSparse(c1, c1);
//			temp2 = Matrix2DUtil.SparseTransposeMultSparse(c2, c2);
			
			// compute A1:
			temp3 = Matrix2DUtil.SparseMultSparse(c1, S);
			A1 = Matrix2DUtil.SparseTransposeMultSparse(dtm, temp3);
			temp3 = null;
			
			// compute A2:
			temp3 = Matrix2DUtil.SparseMultSparse(temp1, S);
			A2 = Matrix2DUtil.SparseTransposeMultSparse(S, temp3);
			temp3 = null;
			
			A2Pos = new ArrayList<DoubleMatrix1D>();
			A2Neg = new ArrayList<DoubleMatrix1D>();
			for (int i = 0; i < A2.size(); ++i) {
				A2Pos.add(new ColtSparseVector(A2.get(i).size()));
				A2Neg.add(new ColtSparseVector(A2.get(i).size()));
			}
			for (int i = 0; i < A2.size(); ++i) {
				IntArrayList indexList = new IntArrayList();
				DoubleArrayList valueList = new DoubleArrayList();
				A2.get(i).getNonZeros(indexList, valueList);
				for (int j = 0; j < indexList.size(); ++j) {
					int index = indexList.get(j);
					double value = valueList.get(j);
					if (value > 0) {
						A2Pos.get(i).set(index, value);
					} else if (value < 0) {
						A2Neg.get(i).set(index, -value);
					}
				}
			}
			
			A2 = null;
			
			temp3 = Matrix2DUtil.SparseMultSparse(c2, A2Neg);
			A2Neg = null;
			temp4 = Matrix2DUtil.SparseMultSparse(c2, A2Pos);
			A2Pos = null;
			
			temp5 = Matrix2DUtil.SparseMultSparse(cCMatNeg, c2);
			temp6 = Matrix2DUtil.SparseMultSparse(cCMatPos, c2);
			
			for (int i = 0; i < c2.size(); ++i) {
				for (int j = 0; j < c2.get(i).size(); ++j) {
					t1 = A1.get(i).get(j);
					if (t1 < 0) {
						t1 = 0;
					}
					t2 = temp3.get(i).get(j);
					t3 = temp5.get(i).get(j);
					t4 = A1.get(i).get(j);
					if (t4 > 0) {
						t4 = 0;
					} else {
						t4 = -t4;
					}
					t5 = temp4.get(i).get(j);
					t6 = temp6.get(i).get(j);
					
					t0 = c2.get(i).get(j);
					
//					t0 = t0 * Math.sqrt((t1 + t2 + t3) / (t4 + t5 + t6 + this.TOLERANCE));
					
					t0 = t0 * Math.sqrt( (t1 + t2 + 1*Math.exp (0 - 1/t3)) / 
							(t4 + t5 + 1*Math.exp (0 - 1/t6) + this.TOLERANCE) );
					
					if (t0 > 1) {
						t0 = 1;
					}
					
					c2.get(i).set(j, t0);
				}
			}
			A1 = null;
			temp3 = null;
			temp4 = null;
			temp5 = null;
			temp6 = null;
			
			temp1 = null;
			temp2 = null;
			
			fullGC(); 
			
			// Compute cost
			cost = 0.0;
			temp1 = Matrix2DUtil.SparseMultSparseTranspose(S, c2);
//			temp2 = Matrix2DUtil.SparseMultSparse(c1, temp1);
//			temp1 = null;
//			for (int i = 0; i < dtm.size(); ++i) {
//				for (int j = 0; j < dtm.get(i).size(); ++j) {
//					cost += Math.pow((dtm.get(i).get(j) - temp2.get(i).get(j)), 2);
//				}
//			}
//			temp2 = null;
			List<DoubleMatrix1D> temp1T = Matrix2DUtil.getSparseTranspose(temp1);
			for (int i = 0; i < dtm.size(); ++i) {
				for (int j = 0; j < dtm.get(i).size(); ++j) {
					double value = Matrix2DUtil.product(c1.get(i), temp1T.get(j));
					cost += Math.pow((dtm.get(i).get(j) - value), 2);
				}
			}
			temp1 = null;
			temp1T = null;
			
			temp1 = Matrix2DUtil.SparseMultSparse(rCMat, c1);
//			temp2 = Matrix2DUtil.SparseTransposeMultSparse(c1, temp1);
//			temp1 = null;
//			for (int i = 0; i < temp2.size(); ++i) {
//				cost += Math.pow(temp2.get(i).get(i), 2);
//			}
//			temp2 = null;
			List<DoubleMatrix1D> c1T = Matrix2DUtil.getSparseTranspose(c1);
			temp1T = Matrix2DUtil.getSparseTranspose(temp1);
			for (int i = 0; i < c1T.size(); ++i) {
				double value = Matrix2DUtil.product(c1T.get(i), temp1T.get(i));
				cost += Math.pow(value, 1);
			}
			c1T = null;
			temp1T = null;
			temp1 = null;
			
			temp3 = Matrix2DUtil.SparseMultSparse(cCMat, c2);
//			temp4 = Matrix2DUtil.SparseTransposeMultSparse(c2, temp3);
//			temp3 = null;
//			for (int i = 0; i < temp4.size(); ++i) {
//				cost += Math.pow(temp4.get(i).get(i), 2);
//			}
//			temp4 = null;
			List<DoubleMatrix1D> c2T = Matrix2DUtil.getSparseTranspose(c2);
			List<DoubleMatrix1D> temp3T = Matrix2DUtil.getSparseTranspose(temp3);
			for (int i = 0; i < c2T.size(); ++i) {
				double value = Matrix2DUtil.product(c2T.get(i), temp3T.get(i));
				cost += Math.pow(value, 1);
			}
			c2T = null;
			temp3T = null;
			temp3 = null;
			
			cost = Math.abs(cost);
			
			delta = Math.abs(oldCost - cost) / (oldCost + this.TOLERANCE);
			
			memberC1 = Matrix2DUtil.matrixMax(c1, 2);		
			memberC2 = Matrix2DUtil.matrixMax(c2, 2);		
			
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
////		double[][] feature = {
////				  {0.03, 0.05, 0.05, 0, 0, 0},
////				  {0.05, 0.03, 0.05, 0, 0, 0},
////				  {0.03, 0.03, 0.03, 0.05, 0.04, 0.05},
////				  {0, 0, 0, 0.05, 0.05, 0.04},
////				  {0.04, 0.04, 0.03, 0.03, 0.04, 0.04},
////				  {0.03, 0.04, 0.03, 0.03, 0.04, 0.03},
////				  };	
//		ConstraintSemiTriNMF sstnmf = new ConstraintSemiTriNMF(feature, 3, 2);
//		sstnmf.setRandomSeed(1);
//		int[] rlabels = {0, 0, 1, 1, 2, 2};
//		sstnmf.setRowClusterLabels(rlabels);
//		int[] clabels = {0, 0, 0, 1, 1, 1};
//		sstnmf.setColumnClusterLabels(clabels);
//		sstnmf.estimate();
//		int[] rlabel = sstnmf.getRowClusterLabels();
//		int[] clabel = sstnmf.getColumnClusterLabels();
//		for (int i = 0; i < rlabel.length; ++i) {
//			System.out.print(rlabel[i] + " ");
//		}
//		System.out.println("");
//		for (int i = 0; i < rlabel.length; ++i) {
//			System.out.print(clabel[i] + " ");
//		}
//		System.out.println("");
		
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
		ConstraintSemiTriNMF sstnmf = new ConstraintSemiTriNMF(feature, 3, 2, 
				rowMust, rowCannot, colMust, colCannot);
		sstnmf.estimate();
		int[] rlabel = sstnmf.getRowClusterLabels();
		int[] clabel = sstnmf.getColumnClusterLabels();
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

