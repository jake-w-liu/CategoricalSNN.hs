#include <cstdlib>

#include <verilated.h>

#include "Vsnn32to16to8.h"

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);

  Vsnn32to16to8 *top = new Vsnn32to16to8;

  while(!Verilated::gotFinish()) {
    top->eval();
  }

  top->final();

  delete top;

  return EXIT_SUCCESS;
}

