# /* Copyright Benjamin Welton 2015 */

MAIN_DIR = /ccs/home/welton/repo/cuda_dclust

all: dbscan 
	echo "built all"

dbscan:
	cd $(MAIN_DIR)/src/dbscan ; \
	$(MAKE) -f Makefile all ;

install: dbscan_install

dbscan_install:
	cd $(MAIN_DIR)/src/dbscan ; \
	$(MAKE) -f Makefile install ;
