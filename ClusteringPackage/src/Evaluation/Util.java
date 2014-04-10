package Evaluation;


import java.util.HashMap;
import java.util.Map;

import cern.colt.map.OpenIntIntHashMap;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;

public class Util{
	static Algebra alg = new Algebra();
	public static int NormalizedLabel(int[] nlabel, final Object[] label){
		Map<Object, Integer> map2Index = new HashMap<Object, Integer>();
		int K = 0;
		for(int i = 0; i < label.length; i++){
			if(map2Index.containsKey(label[i]) == false){
				map2Index.put(label[i], K++);
			}
		}
		for(int i = 0; i < nlabel.length; i++){
			nlabel[i] = map2Index.get(label[i]);
		}
		return K;
	}
	public static int NormalizeLabel(int[] nlabel, final int[] label){	
		OpenIntIntHashMap map2Index = new OpenIntIntHashMap();
		int K = 0;
		for(int i = 0; i < label.length; i++){
			if(map2Index.containsKey(label[i]) == false){
				map2Index.put(label[i], K++);
			}
		}
		for(int i = 0; i < nlabel.length; i++){
			nlabel[i] = map2Index.get(label[i]);
		}
		return K;
	}
	public static DoubleMatrix2D LabelSequence2OneInKMat(int[] label, int K){
		int n = label.length;
		SparseDoubleMatrix2D mat = new SparseDoubleMatrix2D(n, K);
		for(int i = 0; i < n; i++){
			mat.setQuick(i, label[i], 1);
		}
		return mat;
	}
	
	public static DoubleMatrix2D ConfuseMat(int[] classy, int[] clustlb){
		int[] nclassy = new int[classy.length];
		int Ky = NormalizeLabel(nclassy, classy);
		int[] nclustlb = new int[clustlb.length];
		int Kc = NormalizeLabel(nclustlb, clustlb);
		
		DoubleMatrix2D cmat = new DenseDoubleMatrix2D(Ky, Kc);		
		DoubleMatrix2D maty = LabelSequence2OneInKMat(nclassy, Ky);
		DoubleMatrix2D matc = LabelSequence2OneInKMat(nclustlb, Kc);
//		System.out.println("maty = " + maty.toString());
//		System.out.println("matc = " + matc.toString());
		DoubleMatrix2D matyt =alg.transpose(maty);
		matyt.zMult(matc, cmat);
		return cmat;
	}
	
	public static void main(String[] arg){
//		int[] y = new int[]{1, 2, 1, 2, 2, 2};
//		int[] c = new int[]{2, 2, 2, 1, 1, 1};
//		DoubleMatrix2D cmat = ConfuseMat(y, c);
//		String str = cmat.toString();
//		System.out.println(str);
//		
		String[] sy = new String[]{"ab", "ab", "cd", "cd"};
		
				
		int[] ic = new int[sy.length];
		int K = NormalizedLabel(ic, sy);
		System.out.println(K);
		return;
	}
}