#include <cstdlib>

#include <verilated.h>

#include "Vsnn4to3to2Flat.h"

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);

  Vsnn4to3to2Flat *top = new Vsnn4to3to2Flat;

  while(!Verilated::gotFinish()) {
    top->eval();
  }

  top->final();

  delete top;

  return EXIT_SUCCESS;
}

