/**
* A simple recommendation engine that builds association rules from the Apriori algorithm.
* This was originally tested on the Netflix movie review dataset
*/
import java.util.*;
import java.io.*;

public class MovieMine
{
	private static final boolean DEBUG=false;
	
	private static final String FILENAME="ml-100k/u.data";
	private static final int NUMMOVIES=1682; // Looked this up
	private static final int NUMUSERS=943; // This too
	
	//private static final String FILENAME="test.txt";
	//private static final int NUMMOVIES=3;
	//private static final int NUMUSERS=10;
	
	// Create tables
	static int [][] positiveTable=new int [NUMUSERS][NUMMOVIES]; // [users][movies]
	static int [][] negativeTable=new int [NUMUSERS][NUMMOVIES]; 
	static int [][] combinedTable=new int [NUMUSERS][2*NUMMOVIES];
		
	public static void main(String [] args)
	{
		if(args.length<7)
		{
			System.out.println("Insufficient arguments -- exiting");
			System.exit(0);
		}
		// Handle command line arguments
		double minSupport= Double.parseDouble(args[1])/100;
		double minConf= Double.parseDouble(args[3])/100;
		int maxMovies= Integer.parseInt(args[5]);
		if(maxMovies <=0)
		{
			System.out.println("Please enter a positive integer. Exiting.");
			System.exit(0);
		}
		String whichTable=args[6];
		
		transformData(); // Reads in u.data
		
		// Figure out which table
		if(whichTable.equals("-pos"))
		{
			ArrayList<MovieTuple> unigrams = aprioriUnigrams(positiveTable, minSupport,minConf,maxMovies);
			//System.out.println(unigrams);
			if(maxMovies > 1)
			{
				ArrayList<MovieTuple> grams = aprioriNGrams(positiveTable, unigrams,minSupport,minConf,maxMovies);
				//System.out.println(grams);
				makeRules(grams,positiveTable, minConf);
			}
			else
			{
				System.out.println(unigrams);
			}
		}
		else if(whichTable.equals("-neg"))
		{
			ArrayList<MovieTuple> unigrams = aprioriUnigrams(negativeTable, minSupport,minConf,maxMovies);
			//System.out.println(unigrams);
			if(maxMovies > 1)
			{
				ArrayList<MovieTuple> grams = aprioriNGrams(negativeTable, unigrams,minSupport,minConf,maxMovies);
				//System.out.println(grams);
				makeRules(grams, negativeTable, minConf);
			}
			else
			{
				System.out.println(unigrams);
			}
		}
		else if(whichTable.equals("-combo"))
		{
			ArrayList<MovieTuple> unigrams=comboAprioriUnigrams(minSupport,minConf,maxMovies);
			if(maxMovies > 1)
			{
				ArrayList<MovieTuple> grams = comboAprioriNGrams(unigrams,minSupport,minConf,maxMovies);
				//System.out.println(grams);
				comboMakeRules(grams, minConf);
			}
			else
			{
				System.out.println(unigrams);
			}
		}
		else
		{
			System.out.println("Invalid argument, exiting");
			System.exit(0);
		}
		
		
	}
	
	private static double support(Collection<Movie> movies, int [][] table) 
	{
		int currentSupport=0;
		// check support
		for(int user=0;user<NUMUSERS;user++)
		{
			boolean supported = true;
			
			for(Movie i: movies) 
			{
				if(table[user][i.movieNum-1]!=1)
				{
					 supported=false;
					 break;
				}
			 }
			 if(supported) 
			 {
				currentSupport++;
			 }
		}
		return (currentSupport/(double)NUMUSERS);
	}
	
	private static void makeRules(ArrayList<MovieTuple> ngrams, int [][] table, double minConfidence) 
	{
		for(MovieTuple m: ngrams) 
		{
			ArrayList<Movie> movies = new ArrayList<Movie>(m.movieName);
			
			// For 2^n subsets in the power set, left shift is = to *2^n
			// In the resulting binary string for partition, each bit represents
			// whether the corresponding movie should be on the right or left hand
			// side of an association rule
			// We skip 1 and 2^n to avoid empty lhs and rhs rules
			for(int partition=1;partition<((1<<movies.size())-1);partition++) 
			{
				HashSet<Movie> lhs = new HashSet<Movie>();
				HashSet<Movie> rhs = new HashSet<Movie>();
				
				for(int i=0;i<movies.size();i++) 
				{
					// Getting the ith bit
					int b = (partition >> i) & 1;
					if(b == 0) // Add movie [i] to the lhs
					{
						lhs.add(movies.get(i));
					}
					else // add to the right
					{
						rhs.add(movies.get(i));
					}
				}
								
				double conf = support(movies, table) / support(lhs, table);

				if(conf >= minConfidence) 
				{	
					for(Movie i: lhs) 
					{
						System.out.print(i + " ");
					}
					System.out.print("=> ");
					for(Movie i: rhs) 
					{
						System.out.print(i + " ");
					}
					System.out.println();
				}
			}
		}
	}
	
	private static void comboMakeRules(ArrayList<MovieTuple>grams, double minConf)
	{
		for(MovieTuple m: grams) 
		{
			ArrayList<Movie> movies = new ArrayList<Movie>(m.movieName);
			for(int partition=1;partition<((1<<movies.size())-1);partition++) 
			{
				HashSet<Movie> lhs = new HashSet<Movie>();
				HashSet<Movie> rhs = new HashSet<Movie>();
				
				for(int i=0;i<movies.size();i++) 
				{
					int b = (partition >> i) & 1;
					if(b == 0) 
					{
						lhs.add(movies.get(i));
					}
					else
					{
						rhs.add(movies.get(i));
					}
				}
								
				double conf = comboSupport(movies) / comboSupport(lhs);

				if(conf >= minConf) 
				{	
					for(Movie i: lhs) 
					{
						System.out.print(i + " ");
					}
					System.out.print("=> ");
					for(Movie i: rhs) 
					{
						System.out.print(i + " ");
					}
					System.out.println();
				}
			}
		}
	
	}
	
	
	/*-------------------------------------------------------
			
			Algorithm L, Section 7.2.1.3 TAOCP 4A, Knuth.
		
	---------------------------------------------------------*/
	private static class SubsetGenerator 
	{
		private int [] c;
		private int n;
		private int t;
		private boolean stop;
		
		public  SubsetGenerator (int n, int t)
		{
			this.n = n;
			this.t = t;
			stop = false;
			
			c = new int [t+2]; 
			
			for(int j=0;j<t;j++)
			{
				c[j]=j;
			}
			
			c[t]=n;
			c[t+1]=0;
		}
		
		public int[] next()
		{
			int [] result = (int [])c.clone();
			if(stop) 
			{
				return null;
			}
			
			int j=0; 
			
			while(c[j]==(c[j+1]-1))
			{
				c[j]=j;
				j++;
			}
			
			if(j>=t)
			{
				stop = true;
			} 
			
			c[j]=c[j]+1;
			
			return result;
		}
	}
	
	// Returns true if a particular rule should NOT be generated 
	// due to one of its subrules being unsupported
	public static boolean prune(HashSet<Movie> merge, ArrayList<MovieTuple> nMinusOneGrams) 
	{
		int n = merge.size();
		
		Movie [] ngram = new Movie[merge.size()];
		merge.toArray(ngram);
		
		SubsetGenerator subsetGenerator = new SubsetGenerator(n,n-1); 
		//For each subset we generate (which is an array of indices to use from ngram)
		int [] subset=subsetGenerator.next();
		HashSet <Movie> movies=new HashSet <Movie>();
		
		while(subset!=null)
		{
			movies.clear();
			
			for(int i=0;i<(n-1);i++)
			{
				movies.add(ngram[subset[i]]);
			}
			boolean found=false;
			
			//Look for those n-1 movies in the nMinusOneGrams array to see if it is supported
			for(MovieTuple m: nMinusOneGrams)
			{
				if(m.movieName.equals(movies))
				{
					found=true;
					break;
				}	
			}
			if(!found)
			{
				return true;
			}
			subset=subsetGenerator.next(); // Gives subset of elements 
		}
		//if it is not found
		return false;
		
	}
		
	// Apriori for N-Grams, N > 1
	public static ArrayList<MovieTuple> aprioriNGrams(int [][]table, ArrayList<MovieTuple> minusOneGrams,double support, double confidence, int maxMovies)
	{
		ArrayList <MovieTuple> ngrams=null;
		for(int gramSize = 2; gramSize <= maxMovies; gramSize++) 
		{
			if(DEBUG) 
			{
				System.out.println("Currently examining " + minusOneGrams.size() + " " + (gramSize-1) + "-grams" );
			}

			ngrams=new ArrayList<MovieTuple>(); 
			for(int movie1=0;movie1<minusOneGrams.size();movie1++)
			{
				for(int movie2=movie1+1;movie2<minusOneGrams.size();movie2++)
				{
					if(movie2!=movie1) // Skip if they're the same movie 
					{
						MovieTuple leftSide=minusOneGrams.get(movie1); // I picked the names based on how I think
						MovieTuple rightSide=minusOneGrams.get(movie2); // about association rules. Nitpick nitpick.
						
						HashSet<Movie> merge = new HashSet<Movie>();
						merge.addAll(leftSide.movieName);
						merge.addAll(rightSide.movieName);
						
						if(ngrams.contains(new MovieTuple(merge,0)))
						{
							continue;
						}
						
						//System.out.println("Merging: " + merge);
						
						if(merge.size() > gramSize) 
						{
							continue;
						}
						
						if(gramSize > 2 && prune(merge, minusOneGrams) == true) 
						{
							continue;
						}
						
						double sup = support(merge, table);
						
						if(sup>=support) // Both movies are over minSupport
						{
							// create a new tuple for both movies, add it to ngrams
							ngrams.add(new MovieTuple(merge,sup));
						}
					}
				}
			}
			if(ngrams.size() < 1) 
			{
				return minusOneGrams;
			}
			minusOneGrams = ngrams;
		}
		return ngrams;
	}
	
	// Apriori algorithm for a unigram
	public static ArrayList<MovieTuple> aprioriUnigrams(int [][] table, double support, double confidence, int maxMovies)
	{
		int currentSupport=0; // The support for the current movie we are examining
		ArrayList<MovieTuple> unigrams = new ArrayList<MovieTuple>();
		
		// Find unigrams over support limit
		// For each move movie, count the  reviews
		for(int movie=0;movie<NUMMOVIES;movie++)
		{
			currentSupport=0;
			// iterate over users
			for(int user=0;user<NUMUSERS;user++)
			{
				if(table[user][movie]==1)
				{
					currentSupport++;
				}
			}
			double sup = (currentSupport/(double)NUMUSERS);
			if(sup>=support) // Movie is over minSupport
			{
				HashSet <Movie> name=new HashSet <Movie> ();
				name.add(new Movie(movie+1,(table==positiveTable)?0:1));
				MovieTuple currMovie=new MovieTuple(name,sup);
				unigrams.add(currMovie); 
			}
		}
		return unigrams;
	}
	
	// Finds the support for items in the combined table
	private static double comboSupport(Collection<Movie> movies) 
	{
		int currentSupport=0;
		// check support
		for(int user=0;user<NUMUSERS;user++)
		{
			boolean supported = true;
			
			for(Movie i: movies) 
			{
				if(combinedTable[user][(i.movieNum-1)*2+i.table]!=1) //table is either 0 or 1
				{
					 supported=false;
					 break;
				}
			 }
			 if(supported) 
			 {
				currentSupport++;
			 }
		}
		return (currentSupport/(double)NUMUSERS);
	}
	
	// Yep
	public static ArrayList<MovieTuple> comboAprioriNGrams(ArrayList<MovieTuple> minusOneGrams,double support, double confidence, int maxMovies)
	{
		ArrayList <MovieTuple> ngrams=null;
		for(int gramSize = 2; gramSize <= maxMovies; gramSize++) 
		{
			if(DEBUG) 
			{
				System.out.println("Currently examining " + minusOneGrams.size() + " " + (gramSize-1) + "-grams" );
			}

			ngrams=new ArrayList<MovieTuple>(); 
			for(int movie1=0;movie1<minusOneGrams.size();movie1++)
			{
				for(int movie2=movie1+1;movie2<minusOneGrams.size();movie2++)
				{
					if(movie2!=movie1) // Skip if they're the same movie 
					{
						MovieTuple leftSide=minusOneGrams.get(movie1); // I picked the names based on how I think
						MovieTuple rightSide=minusOneGrams.get(movie2); // about association rules. Nitpick nitpick.
						
						HashSet<Movie> merge = new HashSet<Movie>();
						merge.addAll(leftSide.movieName);
						merge.addAll(rightSide.movieName);
						
						if(ngrams.contains(new MovieTuple(merge,0)))
						{
							continue;
						}
						
						//System.out.println("Merging: " + merge);
						
						if(merge.size() > gramSize) 
						{
							continue;
						}
						
						if(gramSize > 2 && prune(merge, minusOneGrams) == true) 
						{
							continue;
						}
						
						// check support
						double sup = comboSupport(merge);
						
				
						if(sup>=support) // Both movies are over minSupport
						{
							// create a new tuple for both movies, add it to ngrams
							ngrams.add(new MovieTuple(merge,sup));
						}
					}
				}
			}
			if(ngrams.size() < 1) 
			{
				return minusOneGrams;
			}
			minusOneGrams = ngrams;
		}
		return ngrams;
	}
	
	// Mmhmm
	public static ArrayList<MovieTuple> comboAprioriUnigrams(double support, double confidence, int maxMovies)
	{
		int posCurrentSupport=0; // The support for the current movie we are examining
		int negCurrentSupport=0;
		ArrayList<MovieTuple> unigrams = new ArrayList<MovieTuple>();
		
		// Find unigrams over support limit
			// For each move movie, count the positive reviews
			for(int movie=0;movie<NUMMOVIES*2;movie+=2)
			{
				posCurrentSupport=0;
				negCurrentSupport=0;
				// iterate over users
				for(int user=0;user<NUMUSERS;user++)
				{
					if(combinedTable[user][movie]==1)
					{
						posCurrentSupport++;
					}
					if(combinedTable[user][movie+1]==1)
					{
						negCurrentSupport++;
					}
				}
				double posSup = (posCurrentSupport/(double)NUMUSERS);
				double negSup = (negCurrentSupport/(double)NUMUSERS);
				
				if(posSup>=support) // Movie is over minSupport
				{
					HashSet <Movie> name=new HashSet <Movie> ();
					name.add(new Movie((movie/2)+1,0));
					MovieTuple currMovie=new MovieTuple(name,posSup);
					unigrams.add(currMovie); 
				}
				if(negSup>=support) // Movie is over minSupport
				{
					HashSet <Movie> name=new HashSet <Movie> ();
					name.add(new Movie((movie/2)+1,1));
					MovieTuple currMovie=new MovieTuple(name,negSup);
					unigrams.add(currMovie); 
				}
			}
		return unigrams;
	}
	
	
	
	// Read in data 
	public static void transformData()
	{
		String line="";
		try
		{
			BufferedReader br = new BufferedReader(new FileReader(FILENAME));
			while ((line = br.readLine()) != null)
			{
				if(line.trim().equals(""))
				{
					continue;
				}
				// Split and save data 
				String[] data=line.split("\t");
				int user=Integer.parseInt(data[0]); // User ID 
				int itemID=Integer.parseInt(data[1]); // Item ID 
				if(Integer.parseInt(data[2])>=4) // Rating 
				{
					positiveTable[user-1][itemID-1]=1;
					negativeTable[user-1][itemID-1]=0;
					combinedTable[user-1][(itemID-1)*2]=1;
					
				}
				else if(Integer.parseInt(data[2])<3)
				{
					positiveTable[user-1][itemID-1]=0;
					negativeTable[user-1][itemID-1]=1;
					combinedTable[user-1][(itemID-1)*2+1]=1;
				}
				
			}	
		}
		catch(IOException e) {e.printStackTrace();}
	}	
	
	// Used for storing data about the movie 
	private static class MovieTuple
	{
		private HashSet<Movie> movieName;
		private double support;
		
		public MovieTuple(HashSet<Movie> movie, double sup)
		{
			movieName=movie;
			support=sup;
		}
		
		public String toString() 
		{
			return movieName.toString();
		}
		
		public boolean equals(Object o) 
		{
			if(!(o instanceof MovieTuple)) 
			{
				return false;
			}
			MovieTuple m = (MovieTuple)o;
			return m.movieName.equals(this.movieName);
		}
	}
	
	// For distinguishing between tables, and formatting
	private static class Movie
	{
		private int movieNum;
		private int table;
		
		public Movie(int ID, int tableNum)
		{
			movieNum=ID;
			table=tableNum;
		}
		
		public String toString() 
		{
			if(table==0)
			{
				return "Pos("+movieNum+")";
			}
			else
			{
				return "Neg("+movieNum+")";
			}
		}
		
		public boolean equals(Object o)
		{
			if(!(o instanceof Movie)) 
			{
				return false;
			}
			Movie m = (Movie)o;
			return (m.movieNum == this.movieNum) && (m.table == this.table);
		}
		
		public int hashCode()
		{
			return this.toString().hashCode();
		}
	}
	
}
