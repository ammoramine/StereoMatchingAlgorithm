# définition des cibles particulières
.PHONY: clean, mrproper
 
# désactivation des règles implicites
.SUFFIXES:

# objets sources
SRC= $(wildcard *.cpp)
OBJ= $(SRC:.cpp=.o)
EXRLIB=-lIex -lHalf -lIlmImf #for iio ... but still mysterious for me

CFFLAGS= -W -Wall -ansi -pedantic -g -w

INC=

INC_PARAMS=$(foreach d, $(INC), -I$d)

LINK=

LINK_PARAMS=$(foreach d, $(LINK), -L$d)


# édition de liens

exec: $(OBJ)
	g++ -o $@ $^ iio.o -g  `pkg-config opencv --libs` -ljpeg -ltiff -lpng $(EXRLIB) -lpthread
# assemblage

%.o: %.cpp %.h
	g++ -o $@ -c $< $(CFFLAGS) `pkg-config opencv --cflags` -std=c++11 
iio.o: iio.c iio.h
	gcc -o iio.o -c iio.c -g -I$CPATH -DI_CAN_HAS_LINUX -DIIO_SHOW_DEBUG_MESSAGES -D_GNU_SOURCE

main.o : main.cpp
	g++ -o $@ -c $< $(CFFLAGS) `pkg-config opencv --cflags` -std=c++11
# édition de liens
	
clean:
	rm -rf *.o
mrproper: clean
	rm -rf exec
