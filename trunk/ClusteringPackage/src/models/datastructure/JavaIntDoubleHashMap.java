package models.datastructure;

import java.util.HashMap;
import java.util.Set;

import cern.colt.function.IntProcedure;
import cern.colt.map.AbstractIntDoubleMap;


public class JavaIntDoubleHashMap extends AbstractIntDoubleMap {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	/**
	 * 
	 */
//	private static final long serialVersionUID = 1L;
	
	
	HashMap<Integer, Double> hashmap = new HashMap<Integer, Double>();

	@Override
	public boolean forEachKey(IntProcedure procedure) {
		Set<Integer> keys = hashmap.keySet();
		for (Integer key : keys) {
			if (! procedure.apply(key)) 
				return false;
		}
		return true;
	}

	@Override
	public double get(int key) {
		if (hashmap.containsKey(key) == false) {
//			System.out.println("what is the problem?");
			return 0;
		}
		return hashmap.get(key);
	}

	@Override
	public boolean put(int key, double value) {
		hashmap.put(key, value);
		if (hashmap.containsKey(key)) {
			return false;
		} else {
			return true;
		}
	}

	@Override
	public boolean removeKey(int key) {
		if (hashmap.containsKey(key)) {
			hashmap.remove(key);
			return true;
		} else {
			return false;
		}
	}

	@Override
	public void clear() {
		hashmap.clear();
	}

}
