package Evaluation;

import cern.colt.matrix.DoubleMatrix2D;

public class Evaluators{
	
	/**
	 * 
	 * @param cmat, nY x nK
	 * @return NMI
	 */
	public static double NormalizedMutualInfo(DoubleMatrix2D cmat){
		double n = cmat.zSum();
		double[][] dcmat = cmat.toArray();
		double[] nk = sum(dcmat, 1);
		double[] ny = sum(dcmat, 2);
		double Iky = 0;//mutual information I(Y,K)
		final double EPS = 10e-20;
		for(int i = 0; i < ny.length; i++){
			for(int j = 0; j < nk.length; j++){
				if(dcmat[i][j] > EPS){// =0
					Iky += dcmat[i][j] / n * Math.log(n * dcmat[i][j] / (nk[j] * ny[i]));
				}
			}
		}
		double Hk = 0;
		for(int i = 0; i < nk.length; i++){
			if(nk[i] > EPS)
				Hk += - nk[i]/n * Math.log(nk[i]/n);
		}
		double Hc = 0;
		for(int i = 0; i < ny.length; i++){
			if(ny[i] > EPS)
				Hc += - ny[i]/n * Math.log(ny[i]/n);
		}

		return 2 * Iky / (Hk + Hc);
	}
	
	public static double NormalizedMutualInfo(int[] classy, int[] clustk){
		return NormalizedMutualInfo(Util.ConfuseMat(classy, clustk));
	}
	public static double Purity(DoubleMatrix2D cmat){
		double n = cmat.zSum();
		double[] maxv = new double[cmat.columns()];
		max(maxv, null, cmat.toArray(), 1);
		return VectorSum(maxv) / n;
	}
	public static double Purity(int[] classy, int[] clustk){
		return Purity(Util.ConfuseMat(classy, clustk));
	}
	private static double Nsel2(double nij){
		if(nij < 2)
			return 0;
		return nij * (nij - 1) / 2.0;
	}
	
	@Deprecated
	public static double AdjustedRandIndex(DoubleMatrix2D cmat){
		double n = cmat.zSum();
		double[][] dcmat = cmat.toArray();
		double[] nk = sum(dcmat, 1);
		double[] ny = sum(dcmat, 2);
		double nij2 = 0;//
		double nij;
		for(int i = 0; i < dcmat.length; i++){
			for(int j = 0; j < dcmat[0].length; j++){
				nij = dcmat[i][j];
				nij2 += Nsel2(nij);
			}
		}
		double ny2 = 0;
		for(int i = 0; i < ny.length; i++){
			ny2 += Nsel2(ny[i]);
		}
		double nk2 = 0;
		for(int i = 0; i < nk.length; i++){
			nk2 += Nsel2(nk[i]);
		}
		double e = (ny2 * nk2) / Nsel2(n);
		return (nij2 - e) / (0.5 * (ny2 + nk2) - e);
	}
	
	@Deprecated
	public static double AdjustedRandIndex(final int[] classy, final int[] clustk){
		return AdjustedRandIndex(Util.ConfuseMat(classy, clustk));
	}
	
	/**
	 * Sum along a dimension of a matrix expressed as an array
	 * @param mat, the matrix
	 * @param direction, 1: along a column, 2: along a row
	 * @return direction=1: res[j] is the summation of the j-th column's elements,
	 * 			direction = 2: res[i] is the summation of the i-th row's elements
	 */
	public static double[] sum(final double[][] mat, int direction){
		double[] res = null;
		int nr = mat.length;
		int nc = mat[0].length;
		if(direction == 1){
			res = new double[nc];
			for(int j = 0; j < nc; j++){
				res[j] = 0;
				for(int i = 0; i < nr; i++){
					res[j] += mat[i][j];
				}
			}
			return res;
		}
		else{
			res = new double[nr];
			for(int i = 0; i < nr; i++){
				res[i] = 0;
				for(int j = 0; j < nc; j++){
					res[i] += mat[i][j];
				}
			}
			return res;
		}
	}
	
	/**
	 * Found the maximum along row or column in a matrix expressed by double[][]
	 * @param mvals, output, the maximum values
	 * @param midxs, output, the index of the maximum in original matrix
	 * @param mat, input, matrix, R x C
	 * @param direction, input, 1: find a maximum along a column, then mvals and midxs are
	 * with length C; 2: find a maximum along a row, then mvals and midxs are with
	 * length R.
	 */
	public static void max(double[] mvals, int[] midxs, final double[][] mat, int direction){
		int mi;
		if(direction == 1){//"min" is taken along a column
			for(int j = 0; j < mat[0].length; j++){//traverse all the columns
				mi = 0;
				for(int i = 0; i < mat.length; i++){
					if(mat[i][j] > mat[mi][j])
					{
						mi = i;
					}
				}
				if(midxs != null){
					midxs[j] = mi;
				}
				if(mvals != null){
					mvals[j] = mat[mi][j];
				}
			}
		}
		else{
			for(int i = 0; i < mat.length; i++){
				mi = 0;
				for(int j = 0; j < mat[0].length; j++){
					if(mat[i][j] > mat[i][mi])
					{
						mi = j;
					}
				}
				if(midxs != null){
					midxs[i] = mi;
				}
				if(mvals != null){
					mvals[i] = mat[i][mi];
				}
			}	
		}
	}
	
	public static double VectorSum(final double[] v){
		double sum = 0;
		for(int i = 0; i < v.length; i++){
			sum += v[i];
		}
		return sum;
	}
	public static void main(String[] arg){
		int[] classy = new int[]{1, 1, 1, 2, 2, 2};
		int[] clustk = new int[]{1, 1, 1, 2, 2, 1};
		double nmi = NormalizedMutualInfo(classy, clustk);
		double purity = Purity(classy, clustk);
		double ari = AdjustedRandIndex(classy, clustk);
		System.out.println("NMI = " + nmi + ", Purity = " + purity + ", ARI = " + ari);
		
	}
}