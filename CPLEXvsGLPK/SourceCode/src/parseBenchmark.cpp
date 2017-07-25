#include "parseBenchmark.h"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
using namespace std;

/*
 * Minimize cx
 * sub to   Ax <=b
 *    x>= 0
 *
 * NetLib Benchmarks
 *
                       PROBLEM SUMMARY TABLE

Name       Rows   Cols   Nonzeros    Bytes  BR      Optimal Value    No.	  GPU
********************** Smaller Dimension LP *********************************************************************************
ADLITTLE     57     97      465       3690        2.2549496316E+05   1	  No (Not Equal)
AFIRO        28     32       88        794       -4.6475314286E+02   2	  Yes (Yes Equal)
BEACONFD    174    262     3476      17475        3.3592485807E+04   3	  Check Cycle
BLEND        75     83      521       3227       -3.0812149846E+01   4	  No
BOEING2     167    143     1339       8761  BR   -3.1501872802E+02   5	  No
BRANDY      221    249     2150      14028        1.5185098965E+03   6	  Check Cycle
E226        224    282     2767      17749       -1.8751929066E+01   7	  No
ISRAEL      175    142     2358      12109       -8.9664482186E+05   8	  -895440 (very close)
KB2          44     41      291       2526  B    -1.7499001299E+03   9	  -0 by both GLPK and GPU
RECIPE       92    180      752       6210  B    -2.6661600000E+02   10	  -0 by both GLPK and GPU
SC105       106    103      281       3307       -5.2202061212E+01   11	  -0 by GPU
SC205       206    203      552       6380       -5.2202061212E+01   12	  -0 by GPU
SC50A        51     48      131       1615       -6.4575077059E+01   13	  -0 by GPU
SC50B        51     48      119       1567       -7.0000000000E+01   14	  -0 by GPU
SCAGR7      130    140      553       4953       -2.3313892548E+06   15	  Check Cycle
SHARE1B     118    225     1182       8380       -7.6589318579E+04   16	  No
SHARE2B      97     79      730       4795       -4.1573224074E+02   17	  -412.462 (very close)
STOCFOR1    118    111      474       4247       -4.1131976219E+04   18	  -0 by GPU
VTP.BASE    199    203      914       8175  B     1.2983146246E+05   19  Check Cycle

****************** SOME MORE TEST *************************************************
BOEING1     351    384     3865      25315  BR   -3.3521356751E+02


************ Large Dimension LP for our GPU implementation **************
25FV47      822   1571    11127      70477        5.5018458883E+03
80BAU3B    2263   9799    29063     298952  B     9.8723216072E+05
AGG         489    163     2541      21865       -3.5991767287E+07
AGG2        517    302     4515      32552       -2.0239252356E+07
AGG3        517    302     4531      32570        1.0312115935E+07
BANDM       306    472     2659      19460       -1.5862801845E+02
BNL1        644   1175     6129      42473        1.9776292856E+03
BNL2       2325   3489    16124     127145        1.8112365404E+03

BORE3D      234    315     1525      13160  B     1.3730803942E+03
CAPRI       272    353     1786      15267  B     2.6900129138E+03
CYCLE      1904   2857    21322     166648  B    -5.2263930249E+00
CZPROB      930   3523    14173      92202  B     2.1851966989E+06
D2Q06C     2172   5167    35674     258038        1.2278423615E+05
D6CUBE      416   6184    43888     167633  B     3.1549166667E+02
DEGEN2      445    534     4449      24657       -1.4351780000E+03
DEGEN3     1504   1818    26230     130252       -9.8729400000E+02
DFL001     6072  12230    41873     353192  B     1.12664E+07 **
ETAMACRO    401    688     2489      21915  B    -7.5571521774E+02
FFFFF800    525    854     6235      39637        5.5567961165E+05
FINNIS      498    614     2714      23847  B     1.7279096547E+05
FIT1D        25   1026    14430      51734  B    -9.1463780924E+03
FIT1P       628   1677    10894      65116  B     9.1463780924E+03
FIT2D        26  10500   138018     482330  B    -6.8464293294E+04
FIT2P      3001  13525    60784     439794  B     6.8464293232E+04
FORPLAN     162    421     4916      25100  BR   -6.6421873953E+02
GANGES     1310   1681     7021      60191  B    -1.0958636356E+05
GFRD-PNC    617   1092     3467      24476  B     6.9022359995E+06
GREENBEA   2393   5405    31499     235711  B    -7.2462405908E+07
GREENBEB   2393   5405    31499     235739  B    -4.3021476065E+06
GROW15      301    645     5665      35041  B    -1.0687094129E+08
GROW22      441    946     8318      50789  B    -1.6083433648E+08
GROW7       141    301     2633      17043  B    -4.7787811815E+07
LOTFI       154    308     1086       6718       -2.5264706062E+01
MAROS       847   1443    10006      65906  B    -5.8063743701E+04
MAROS-R7   3137   9408   151120    4812587        1.4971851665E+06
MODSZK1     688   1620     4158      40908  B     3.2061972906E+02
NESM        663   2923    13988     117828  BR    1.4076073035E+07
PEROLD      626   1376     6026      47486  B    -9.3807580773E+03
PILOT      1442   3652    43220     278593  B    -5.5740430007E+02
PILOT.JA    941   1988    14706      97258  B    -6.1131344111E+03
PILOT.WE    723   2789     9218      79972  B    -2.7201027439E+06
PILOT4      411   1000     5145      40936  B    -2.5811392641E+03
PILOT87    2031   4883    73804     514192  B     3.0171072827E+02
PILOTNOV    976   2172    13129      89779  B    -4.4972761882E+03
QAP8        913   1632     8304 (see NOTES)       2.0350000000E+02
QAP12      3193   8856    44244 (see NOTES)       5.2289435056E+02
QAP15      6331  22275   110700 (see NOTES)       1.0409940410E+03
SCAGR25     472    500     2029      17406       -1.4753433061E+07
SCFXM1      331    457     2612      19078        1.8416759028E+04
SCFXM2      661    914     5229      37079        3.6660261565E+04
SCFXM3      991   1371     7846      53828        5.4901254550E+04
SCORPION    389    358     1708      12186        1.8781248227E+03
SCRS8       491   1169     4029      36760        9.0429998619E+02
SCSD1        78    760     3148      17852        8.6666666743E+00
SCSD6       148   1350     5666      32161        5.0500000078E+01
SCSD8       398   2750    11334      65888        9.0499999993E+02
SCTAP1      301    480     2052      14970        1.4122500000E+03
SCTAP2     1091   1880     8124      57479        1.7248071429E+03
SCTAP3     1481   2480    10734      78688        1.4240000000E+03
SEBA        516   1028     4874      38627  BR    1.5711600000E+04
SHELL       537   1775     4900      38049  B     1.2088253460E+09
SHIP04L     403   2118     8450      57203        1.7933245380E+06
SHIP04S     403   1458     5810      41257        1.7987147004E+06
SHIP08L     779   4283    17085     117083        1.9090552114E+06
SHIP08S     779   2387     9501      70093        1.9200982105E+06
SHIP12L    1152   5427    21597     146753        1.4701879193E+06
SHIP12S    1152   2763    10941      82527        1.4892361344E+06
SIERRA     1228   2036     9252      76627  B     1.5394362184E+07
STAIR       357    467     3857      27405  B    -2.5126695119E+02
STANDATA    360   1075     3038      26135  B     1.2576995000E+03
STANDGUB    362   1184     3147      27836  B     (see NOTES)
STANDMPS    468   1075     3686      29839  B     1.4060175000E+03
STOCFOR2   2158   2031     9492      79845       -3.9024408538E+04
STOCFOR3  16676  15695    74004 (see NOTES)      -3.9976661576E+04
TRUSS      1001   8806    36642 (see NOTES)       4.5881584719E+05
TUFF        334    587     4523      29439  B     2.9214776509E-01
WOOD1P      245   2594    70216     328905        1.4429024116E+00
WOODW      1099   8405    37478     240063        1.3044763331E+00
*
*/

void selectBenchmark(int benchmarkNo, std::string& argvA, std::string&  argvB, std::string& argvC){

	if (benchmarkNo==1){//ADLITTLE  has solution as 2.2549496316E+05
		argvA = "benchmarks/ADLITTLE_A.txt"; argvB = "benchmarks/ADLITTLE_B.txt";	argvC = "benchmarks/ADLITTLE_C.txt";
	} else if (benchmarkNo==2){ //AFIRO has solution as -4.6475314286E+02
		argvA="benchmarks/AFIRO_A.txt";  argvB="benchmarks/AFIRO_B.txt"; argvC="benchmarks/AFIRO_C.txt";
	} else if (benchmarkNo==3){ //BEACONFD has solution as 3.3592485807E+04
		argvA="benchmarks/BEACONFD_A.txt";  argvB="benchmarks/BEACONFD_B.txt"; argvC="benchmarks/BEACONFD_C.txt";
	} else if (benchmarkNo==4){//BLEND  has solution as -3.0812149846E+01
		argvA="benchmarks/BLEND_A.txt";  argvB="benchmarks/BLEND_B.txt"; argvC="benchmarks/BLEND_C.txt";
	} else if (benchmarkNo==5){//BOEING2  has solution as -3.1501872802E+02
		argvA="benchmarks/BOEING2_A.txt";  argvB="benchmarks/BOEING2_B.txt"; argvC="benchmarks/BOEING2_C.txt";
	} else if (benchmarkNo==6){//BRANDY  has solution as 1.5185098965E+03
		argvA="benchmarks/BRANDY_A.txt";  argvB="benchmarks/BRANDY_B.txt"; argvC="benchmarks/BRANDY_C.txt";
	} else if (benchmarkNo==7){//E226  has solution as -1.8751929066E+01
		argvA="benchmarks/E226_A.txt";  argvB="benchmarks/E226_B.txt"; argvC="benchmarks/E226_C.txt";
	} else if (benchmarkNo==8){//ISRAEL  has solution as -8.9664482186E+05
		argvA="benchmarks/ISRAEL_A.txt";  argvB="benchmarks/ISRAEL_B.txt"; argvC="benchmarks/ISRAEL_C.txt";
	} else if (benchmarkNo==9){//KB2  has solution as -1.7499001299E+03
		argvA="benchmarks/KB2_A.txt";  argvB="benchmarks/KB2_B.txt"; argvC="benchmarks/KB2_C.txt";
	} else if (benchmarkNo==10){//RECIPE  has solution as  -2.6661600000E+02
		argvA="benchmarks/RECIPELP_A.txt";  argvB="benchmarks/RECIPELP_B.txt"; argvC="benchmarks/RECIPELP_C.txt";
	} else if (benchmarkNo==11){//SC105  has solution as -5.2202061212E+01
		argvA="benchmarks/SC105_A.txt";  argvB="benchmarks/SC105_B.txt"; argvC="benchmarks/SC105_C.txt";
	} else if (benchmarkNo==12){//SC205  has solution as -5.2202061212E+01
		argvA="benchmarks/SC205_A.txt";  argvB="benchmarks/SC205_B.txt"; argvC="benchmarks/SC205_C.txt";
	} else if (benchmarkNo==13){//SC50A  has solution as -6.4575077059E+01
		argvA="benchmarks/SC50A_A.txt";  argvB="benchmarks/SC50A_B.txt"; argvC="benchmarks/SC50A_C.txt";
	} else if (benchmarkNo==14){//SC50B  has solution as -7.0000000000E+01
		argvA="benchmarks/SC50B_A.txt";  argvB="benchmarks/SC50B_B.txt"; argvC="benchmarks/SC50B_C.txt";
	} else if (benchmarkNo==15){//SCAGR7  has solution as -2.3313892548E+06
		argvA="benchmarks/SCAGR7_A.txt";  argvB="benchmarks/SCAGR7_B.txt"; argvC="benchmarks/SCAGR7_C.txt";
	} else if (benchmarkNo==16){//SHARE1B  has solution as -7.6589318579E+04
		argvA="benchmarks/SHARE1B_A.txt";  argvB="benchmarks/SHARE1B_B.txt"; argvC="benchmarks/SHARE1B_C.txt";
	} else if (benchmarkNo==17){//SHARE2B  has solution as -4.1573224074E+02
		argvA="benchmarks/SHARE2B_A.txt";  argvB="benchmarks/SHARE2B_B.txt"; argvC="benchmarks/SHARE2B_C.txt";
	} else if (benchmarkNo==18){//STOCFOR1   has solution as -4.1131976219E+04
		argvA="benchmarks/STOCFOR1_A.txt";  argvB="benchmarks/STOCFOR1_B.txt"; argvC="benchmarks/STOCFOR1_C.txt";
	} else if (benchmarkNo==19){//VTP-BASE  has solution as 1.2983146246E+05
		argvA="benchmarks/VTP-BASE_A.txt";  argvB="benchmarks/VTP-BASE_B.txt"; argvC="benchmarks/VTP-BASE_C.txt";
	}
}

void parseLP(const char* argvA,const  char* argvB,const  char* argvC, math::matrix<double>& A, std::vector<double>& b, std::vector<double>& c, unsigned int& MaxMinFlag){
	/*std::cout << "Argument 1) Afile -- "<<argvA <<std::endl;
	std::cout << "Argument 2) Bfile -- "<<argvB <<std::endl;
	std::cout << "Argument 3) Cfile -- "<<argvC <<std::endl;*/

	int row = 0, col = 0;
	double field;
	int tot_constraints, tot_variables;
	std::ifstream in(argvA);
	if (in) {
		std::string line1;
		std::getline(in, line1); //Reading First Line for total Constraints and Variables
		std::istringstream row1(line1);
		row1 >> tot_constraints; //Reading First value
		row1 >> tot_variables; //Reading Second value
		std::cout << "Total Constraints =" << tot_constraints << "   ";
		std::cout << "Total Variables =" << tot_variables << std::endl;
		A.resize(tot_constraints, tot_variables);
		b.resize(tot_constraints);	//total rows or constraints same as A
		c.resize(tot_variables);	//total number of variables same as A
		//reading remaining Lines
		row = 0; //First Constraint
		while (std::getline(in, line1)) {
			col = 0;
			std::istringstream row1(line1);
			while (row1 >> field) {
				A(row, col) = field;
				//cout << field <<"\t";
				col++;
			}
			row++; //Next Constraints
			//cout<<"\n";
		}
	}//Reading A matrix is Over
	std::ifstream inB(argvB); //reading Bfile
	if (inB) {
		std::string line1;
		row = 0; //First Constraint
		while (std::getline(inB, line1)) {
			std::istringstream row1(line1);
			row1 >> field;
			b[row] = field;
			//cout << field <<"\t";
			row++; //Next Constraints
			//cout<<"\n";
		}
	}//Reading A matrix is Over
	std::ifstream inC(argvC); //reading Cfile
	if (inC) {
		std::string line1;
		row = 0; //First Constraint
		while (std::getline(inC, line1)) {
			std::istringstream row1(line1);
			row1 >> field;
			c[row] = field;
			//cout << field <<"\t";
			row++; //Next Constraints
			//cout<<"\n";
		}
	}//Reading A matrix is Over

	MaxMinFlag = 1; //Minimize
}
