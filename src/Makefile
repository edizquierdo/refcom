main: main.o CTRNN.o TSearch.o CountingAgent.o random.o
	g++ -pthread -o main main.o CTRNN.o TSearch.o CountingAgent.o random.o
random.o: random.cpp random.h VectorMatrix.h
	g++ -pthread -c -O3 random.cpp
CTRNN.o: CTRNN.cpp random.h CTRNN.h
	g++ -pthread -c -O3 CTRNN.cpp
TSearch.o: TSearch.cpp TSearch.h
	g++ -pthread -c -O3 TSearch.cpp
CountingAgent.o: CountingAgent.cpp CountingAgent.h TSearch.h CTRNN.h random.h VectorMatrix.h
	g++ -pthread -c -O3 CountingAgent.cpp
main.o: main.cpp CTRNN.h CountingAgent.h TSearch.h
	g++ -pthread -c -O3 main.cpp
clean:
	rm *.o main
