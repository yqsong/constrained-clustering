package models.util.random;

import cc.mallet.util.Randoms;

public class RandomSampler {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2708396183850779371L;

	public RandomSampler(){
		super();
	}
	final static int MAX_SAMPLE_SIZE = 1000000;
	static int[] RAND_PERM_INTS = new int[MAX_SAMPLE_SIZE];
//	private static Random m_Rander;
	static{
		for(int i = 0; i < MAX_SAMPLE_SIZE; i++){
			RAND_PERM_INTS[i] = i;
		}
	}
	public synchronized int[] RandPerm(int n){
		int[] vas = new int[n];
		System.arraycopy(RAND_PERM_INTS, 0, vas, 0, n);
		for(int i = n - 1; i >= 0; i--){
			int idx = (int)((i+1) * Math.random());
			if(idx != i){
				int temp = vas[idx];
				vas[idx] = vas[i];
				vas[i] = temp;
			}
		}
		return vas;
	}
	
	public double[] nextGamma(double[] alpha){
		Randoms r = new Randoms();
		//sampling from a gamma distribution with shape alpha and scale 1
		double[] res = new double[alpha.length];
		for(int i = 0; i < alpha.length; i++){
			res[i] = r.nextGamma(alpha[i], 1);
		}
		return res;
	}
	public double[] nextDirichlet(double[] alpha) {
		/**
		 * draw a multinomial from a dirichlet with parameter aa
		 * according to the property that the mean of a set of i.i.d. gamma samples 
		 * conforms to a dirichlet distribution
		 */
		double[] gmspl = nextGamma(alpha);
        double sum = 0;
        for (int i = 0; i < gmspl.length; i++) {
        	sum += gmspl[i];
        }
        for (int i = 0; i < gmspl.length; i++) {
        	gmspl[i] /= sum;
        }
        return gmspl;
    }
	public static int nextMultnomial(double[] pi){
		double sum = 0;
		double rd = Math.random();
		for(int i = 0; i < pi.length; i++){
			sum += pi[i];
			if(rd < sum)
				return i;
		}
		return pi.length - 1;
	}
	public static int[] nextMultnomial(double[] pi, int n){
		/* sampling a multinomial, repeat n times, return a vector with counts in each atom
		 * n: repeat time
		 */
		int[] res = new int[pi.length];
		double cumpp[] = new double[pi.length];
		cumpp[0] = pi[0];
		res[0] = 0;
		for(int i = 1; i < res.length; i++)
		{
			res[i] = 0;
			cumpp[i] = cumpp[i-1] + pi[i];
		}
		for(int r = 0; r < n; r++){
			double rd = Math.random();
			for(int i = 0; i < cumpp.length; i++){
				if(rd < cumpp[i]){
					res[i] += 1;
					break;
				}
			}
		}
		return res;
		
	}
	public static int[] nextMultinomialSeq(double[] pi, int n){
		int[] res = new int[n];
		double cumpp[] = new double[pi.length];
		cumpp[0] = pi[0];
		
		for(int i = 1; i < res.length; i++)
		{
			cumpp[i] = cumpp[i-1] + pi[i];
		}
		for(int r = 0; r < n; r++){
			double rd = Math.random();
			for(int i = 0; i < cumpp.length; i++){
				if(rd < cumpp[i]){
					res[r] =  i;
					break;
				}
			}
		}
		return res;
	}
	
	public double[] nextSphericalGaussian(double[] mu, double sigma2){
		Randoms r = new Randoms();
		double[] res = new double[mu.length];
		for(int i = 0; i < mu.length; i++){
			res[i] = r.nextGaussian(mu[i], sigma2);
		}
		return res;
	}
	
	public static void main(String[] arg){
		RandomSampler rdm = new RandomSampler();
		int[] rpm = rdm.RandPerm(10);
//		System.out.println(VectorOper.ToString(rpm));
	}
}