package Evaluation;

public class ComputeMeanVariance {
	
	public static double computeMean (double[] data) {
		double mean = 0;
		for (int i = 0; i < data.length; ++i) {
			mean += data[i];
		}
		mean /= data.length;
		return mean;
	}
	
	public static double computeStandardDeviation (double[] data, double mean) {
		double var = 0;
		for (int i = 0; i < data.length; ++i) {
			var += (data[i] - mean) * (data[i] - mean);
		}
		var = Math.sqrt(var/(data.length - 1));
		return var;
	}
	
	
	public static double[] computeMean (double[][] data) {
		double[] mean = new double [data[0].length];
		for (int i = 0; i < data.length; ++i) {
			for (int j = 0; j < data[i].length; ++j) {
				if (data[i].length != data[0].length) {
					System.out.println("Error: dimension not match");
					return null;
				}
				mean[j] += data[i][j];
			}
		}
		for (int i = 0; i < mean.length; ++i) {
			mean[i] /= data.length;
		}
		return mean;
	}
	
	public static double[] computeMax (double[][] data) {
		double[] max = new double [data[0].length];
		for (int i = 0; i < data.length; ++i) {
			for (int j = 0; j < data[i].length; ++j) {
				if (data[i].length != data[0].length) {
					System.out.println("Error: dimension not match");
					return null;
				}
				if (max[j] < data[i][j]) {
					max[j] = data[i][j];
				}
			}
		}
		return max;
	}
	
	public static double[] computeStandardDeviation (double[][] data, double[] mean) {
		double[] var = new double [data[0].length];
		if (mean.length != var.length) {
			System.out.println("Error: dimension not match");
			return null;
		}
		for (int i = 0; i < data.length; ++i) {
			for (int j = 0; j < data[i].length; ++j) {
				if (data[i].length != data[0].length) {
					System.out.println("Error: dimension not match");
					break;
				}
				var[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
			}
		}
		for (int i = 0; i < mean.length; ++i) {
			var[i] = Math.sqrt(var[i]/(data.length - 1));
		}
		return var;
	}
	
	public static double[][] computeMean (double[][][] data) {
		double[][] mean = new double [data[0].length][data[0][0].length];
		for (int k = 0; k < data.length; ++k) {
			for (int i = 0; i < data[k].length; ++i) {
				for (int j = 0; j < data[k][i].length; ++j) {
					if (data[k].length != data[0].length) {
						System.out.println("Error: dimension not match");
						return null;
					}
					mean[i][j] += data[k][i][j];
				}
			}
		}
		for (int i = 0; i < mean.length; ++i) {
			for (int j = 0; j < mean[i].length; ++j) {
				mean[i][j] /= data.length;
			}
		}
		return mean;
	}
	
	public static double[][] computeStandardDeviation (double[][][] data, double[][] mean) {
		double[][] var = new double [data[0].length][data[0][0].length];
		if (mean.length != var.length) {
			System.out.println("Error: dimension not match");
			return null;
		}
		for (int k = 0; k < data.length; ++k) {
			for (int i = 0; i < data[k].length; ++i) {
				for (int j = 0; j < data[k][i].length; ++j) {
					if (data[k].length != data[0].length) {
						System.out.println("Error: dimension not match");
						return null;
					}
					var[i][j] += (data[k][i][j] - mean[i][j]) * (data[k][i][j] - mean[i][j]);
				}
			}
		}
		for (int i = 0; i < var.length; ++i) {
			for (int j = 0; j < var[i].length; ++j) {
				var[i][j] = Math.sqrt(var[i][j]/(data.length - 1));
			}
		}
		return var;
	}
	
}
