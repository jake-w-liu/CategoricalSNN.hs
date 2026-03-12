#include <cstdlib>

#include <verilated.h>

#include "Vsnn16to8to4.h"

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);

  Vsnn16to8to4 *top = new Vsnn16to8to4;

  while(!Verilated::gotFinish()) {
    top->eval();
  }

  top->final();

  delete top;

  return EXIT_SUCCESS;
}

