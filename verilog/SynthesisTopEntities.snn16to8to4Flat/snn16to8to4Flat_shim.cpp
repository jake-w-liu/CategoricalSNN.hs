#include <cstdlib>

#include <verilated.h>

#include "Vsnn16to8to4Flat.h"

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);

  Vsnn16to8to4Flat *top = new Vsnn16to8to4Flat;

  while(!Verilated::gotFinish()) {
    top->eval();
  }

  top->final();

  delete top;

  return EXIT_SUCCESS;
}

