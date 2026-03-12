#include <cstdlib>

#include <verilated.h>

#include "Vsnn4to3to2.h"

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);

  Vsnn4to3to2 *top = new Vsnn4to3to2;

  while(!Verilated::gotFinish()) {
    top->eval();
  }

  top->final();

  delete top;

  return EXIT_SUCCESS;
}

