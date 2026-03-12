#include <cstdlib>

#include <verilated.h>

#include "Vsnn12to7to5to3.h"

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);

  Vsnn12to7to5to3 *top = new Vsnn12to7to5to3;

  while(!Verilated::gotFinish()) {
    top->eval();
  }

  top->final();

  delete top;

  return EXIT_SUCCESS;
}

