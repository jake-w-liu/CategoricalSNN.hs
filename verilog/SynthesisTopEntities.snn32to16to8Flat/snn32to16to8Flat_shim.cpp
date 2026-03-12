#include <cstdlib>

#include <verilated.h>

#include "Vsnn32to16to8Flat.h"

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);

  Vsnn32to16to8Flat *top = new Vsnn32to16to8Flat;

  while(!Verilated::gotFinish()) {
    top->eval();
  }

  top->final();

  delete top;

  return EXIT_SUCCESS;
}

