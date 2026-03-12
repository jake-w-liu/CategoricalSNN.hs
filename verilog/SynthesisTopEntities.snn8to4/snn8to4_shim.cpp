#include <cstdlib>

#include <verilated.h>

#include "Vsnn8to4.h"

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);

  Vsnn8to4 *top = new Vsnn8to4;

  while(!Verilated::gotFinish()) {
    top->eval();
  }

  top->final();

  delete top;

  return EXIT_SUCCESS;
}

