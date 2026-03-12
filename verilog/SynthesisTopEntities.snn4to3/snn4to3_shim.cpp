#include <cstdlib>

#include <verilated.h>

#include "Vsnn4to3.h"

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);

  Vsnn4to3 *top = new Vsnn4to3;

  while(!Verilated::gotFinish()) {
    top->eval();
  }

  top->final();

  delete top;

  return EXIT_SUCCESS;
}

