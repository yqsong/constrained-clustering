/**
 * 
 */
package models.datastructure;

/*
 * Note this class will lead to reverse sort (desent order)
 */

@SuppressWarnings("unchecked")
public class IDReverseSorter implements Comparable {
    int id; double value;
    
    public IDReverseSorter (int id, double value) { 
        this.id = id; 
        this.value = value; 
    }
    
    public IDReverseSorter (int id, int value) { 
        this.id = id; 
        this.value = value; 
    }
    
    public final int compareTo (Object obj) {
        if (this.value > ((IDReverseSorter) obj).value)
            return -1;
        else if (this.value == ((IDReverseSorter) obj).value)
            return 0;
        else return 1;
    }
    
    public int getID() {
        return this.id;
    }
    
    public double getValue() {
        return this.value;
    }

    public void set(int id, double value) { 
        this.id = id; 
        this.value = value; 
    }

        

}
