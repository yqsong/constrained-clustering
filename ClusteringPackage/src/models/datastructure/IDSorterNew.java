
package models.datastructure;


@SuppressWarnings("unchecked")
public class IDSorterNew implements Comparable<IDSorterNew> {

    int id; double value;
    
    public IDSorterNew (int id, double value) { 
        this.id = id; 
        this.value = value; 
    }
    
    public IDSorterNew (int id, int value) { 
        this.id = id; 
        this.value = value; 
    }
    
    public final int compareTo (IDSorterNew obj) {
        if (this.value > obj.value)
            return 1;
        else if (this.value == obj.value)
            return 0;
        else return -1;
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