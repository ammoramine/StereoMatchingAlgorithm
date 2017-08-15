# définition des cibles particulières
.PHONY: clean, mrproper
 
# désactivation des règles implicites
.SUFFIXES:

# objets sources
SRC= $(wildcard *.cpp)
OBJ= $(SRC:.cpp=.o)


CFFLAGS= -W -Wall -ansi -pedantic -g -w

INC=

INC_PARAMS=$(foreach d, $(INC), -I$d)

LINK=

LINK_PARAMS=$(foreach d, $(LINK), -L$d)


# édition de liens

exec: $(OBJ)
	#g++ -o $@ $^ $(LINK_PARAMS) 
	g++ -o $@ $^  -g  `pkg-config opencv --libs`
# assemblage

# matching_algorithm.o: computeRPCFromCub.cpp
# 	g++ -o $@ -c $< $(CFFLAGS) $(INC_PARAMS)

%.o: %.cpp
	# g++ -o hello.o -c hello.cpp 
	g++ -o $@ -c $< $(CFFLAGS) `pkg-config opencv --cflags`
# édition de liens
	
clean:
	rm -rf *.o
mrproper: clean
	rm -rf exec
