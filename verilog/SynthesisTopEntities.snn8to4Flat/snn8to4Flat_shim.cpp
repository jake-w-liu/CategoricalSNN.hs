#include <cstdlib>

#include <verilated.h>

#include "Vsnn8to4Flat.h"

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);

  Vsnn8to4Flat *top = new Vsnn8to4Flat;

  while(!Verilated::gotFinish()) {
    top->eval();
  }

  top->final();

  delete top;

  return EXIT_SUCCESS;
}

