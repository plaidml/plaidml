#include <algorithm>

#include "tile/base/shape.h"
#include "tile/lang/bound.h"
#include "tile/lang/compile.h"
#include "tile/lang/defract.h"
#include "tile/lang/flat.h"
#include "tile/lang/gen_contract.h"
#include "tile/lang/generate.h"
#include "tile/lang/matrix.h"
#include "tile/lang/ops.h"
#include "tile/lang/parser.h"
#include "tile/lang/reduce.h"
#include "tile/lang/simulate.h"

#include "base/util/catch.h"
#include "base/util/logging.h"

namespace vertexai {
namespace tile {
namespace lang {

TEST_CASE("Evil convolution", "[simulate]") {
  Parser p;
  auto c = p.ParseContraction("O[n,co,x,y] = +(I[n,ci,x+i,y+j] * K[i,j,co,ci])");
  Tensor<float, 4> O(boost::extents[1][3][2][3]);
  Tensor<float, 4> O2(boost::extents[1][3][2][3]);
  Tensor<float, 4> I(boost::extents[1][3][2][3]);
  Tensor<float, 4> K(boost::extents[2][2][3][3]);
  for (size_t i = 0; i < 1 * 3 * 2 * 3; i++) {
    I.origin()[i] = i;
    O.origin()[i] = 0;
  }
  for (size_t i = 0; i < 2 * 2 * 3 * 3; i++) {
    K.origin()[i] = i;
  }
  std::vector<TensorShape> shapes = {ShapeOf(O), ShapeOf(I), ShapeOf(K)};
  FlatContraction fc = Compile(c, shapes);
  Execute(fc, O, I, K);
  for (size_t x = 0; x < 2; x++) {
    for (size_t y = 0; y < 3; y++) {
      for (size_t co = 0; co < 3; co++) {
        for (size_t i = 0; i < 2; i++) {
          for (size_t j = 0; j < 2; j++) {
            for (size_t ci = 0; ci < 3; ci++) {
              if (x + i >= 2 || y + j >= 3) {
                continue;
              }
              O2[0][co][x][y] += I[0][ci][x + i][y + j] * K[i][j][co][ci];
            }
          }
        }
      }
    }
  }
  for (size_t i = 0; i < 1 * 3 * 2 * 3; i++) {
    IVLOG(3, O.origin()[i] << " vs " << O2.origin()[i]);
    REQUIRE(O.origin()[i] == O2.origin()[i]);
  }
}

TEST_CASE("Example function used in maxpool", "[maxpool][keras]") {
  Parser p;
  Contraction c = p.ParseContraction("O[n,x,y,c] = >(I[n, 2*x+i, 2*y+j, c]), i<2, j<2");

  Tensor<float, 4> O(boost::extents[3][2][3][3]);
  Tensor<float, 4> Omanual(boost::extents[3][2][3][3]);
  Tensor<float, 4> I(boost::extents[3][4][6][3]);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 2; j++) {
      for (size_t k = 0; k < 3; k++) {
        for (size_t l = 0; l < 3; l++) {
          O[i][j][k][l] = 0;
        }
      }
    }
  }

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t k = 0; k < 6; k++) {
        for (size_t l = 0; l < 3; l++) {
          if (i == 0) {
            I[i][j][k][l] = 0;
          } else if (i == 1) {
            if (j == 1 || k == 2) {
              I[i][j][k][l] = 1;
            } else {
              I[i][j][k][l] = 0;
            }
          } else if (i == 2) {
            if (l == 0) {
              I[i][j][k][l] = 1;
            } else if (l == 1) {
              I[i][j][k][l] = ((j + k) % 6 == 0) ? j + 4 : 0;
            } else {
              I[i][j][k][l] = 0;
            }
          }
        }
      }
    }
  }

  std::vector<TensorShape> shapes = {ShapeOf(O), ShapeOf(I)};
  FlatContraction fc = Compile(c, shapes);
  Execute(fc, O, I);

  for (size_t n = 0; n < 3; n++) {
    for (size_t x = 0; x < 2; x++) {
      for (size_t y = 0; y < 3; y++) {
        for (size_t c = 0; c < 3; c++) {
          Omanual[n][x][y][c] = 0;
          for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 2; j++) {
              Omanual[n][x][y][c] = std::max(Omanual[n][x][y][c], I[n][2 * x + i][2 * y + j][c]);
            }
          }
        }
      }
    }
  }

  for (size_t l = 0; l < 3; l++) {
    for (size_t i = 0; i < 3; i++) {
      for (size_t k = 0; k < 3; k++) {
        for (size_t j = 0; j < 2; j++) {
          IVLOG(3, std::to_string(O[i][j][k][l])
                       << " from TILE vs " << std::to_string(Omanual[i][j][k][l]) << " expected; at (" << i << ", " << j
                       << ", " << k << ", " << l << ")");
          REQUIRE(O[i][j][k][l] == Omanual[i][j][k][l]);
        }
      }
    }
  }
}

TEST_CASE("Example function used in conv2d", "[conv2d][keras]") {
  Parser p;
  Contraction c = p.ParseContraction("O[n, x, y, co] = +(I[n, x + i - 1, y + j - 1, ci]*K[i, j, ci, co])");

  std::stringstream strstream;
  strstream << "c.specs: size " << std::to_string(c.specs.size()) << ", contents: " << to_string(c.specs) << "\n";
  IVLOG(4, strstream.str());
  strstream << "c.constraints: " << std::to_string(c.constraints) << "\n";
  IVLOG(4, strstream.str());

  Tensor<float, 4> O(boost::extents[5][4][4][4]);
  Tensor<float, 4> I(boost::extents[5][4][4][3]);
  Tensor<float, 4> K(boost::extents[3][3][3][4]);
  for (size_t i = 0; i < 5; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t k = 0; k < 4; k++) {
        for (size_t l = 0; l < 3; l++) {
          O[i][j][k][l] = 0;
        }
      }
    }
  }
  for (size_t i = 0; i < 5; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t k = 0; k < 4; k++) {
        for (size_t l = 0; l < 3; l++) {
          float value;
          switch (i) {
            case 0:
              value = 128;
              break;
            case 1:
              value = (j % 2) ? 255 : 0;
              break;
            case 2:
              value = (j < 2 && k < 2) ? 0 : 255;
              break;
            case 3:
              value = 255;
              break;
            case 4:
              value = 0;
              break;
            default:
              throw std::runtime_error("unexpected i");
          }
          I[i][j][k][l] = value;
        }
      }
    }
  }
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      for (size_t k = 0; k < 3; k++) {
        for (size_t l = 0; l < 4; l++) {
          float value;
          switch (l) {
            case 0:
              value = 1;
              break;
            case 1:
              value = 0;
              break;
            case 2:
              value = k ? 0 : 1;
              break;
            case 3:
              value = k ? 0 : 1.0 / 255.0;
              break;
          }
          K[i][j][k][l] = value;
        }
      }
    }
  }
  std::vector<TensorShape> shapes = {ShapeOf(O), ShapeOf(I), ShapeOf(K)};
  FlatContraction fc = Compile(c, shapes);
  Execute(fc, O, I, K);

  // Putting constructionBuffer in [image][col][row] order to make it *appear*
  // the way matrices normally do.
  // It'll be transposed to be moved into ExpectedO
  float constructionBuffer[5][4][4] = {{{(128.0 / 255) * 4, (128.0 / 255) * 6, (128.0 / 255) * 6, (128.0 / 255) * 4},
                                        {(128.0 / 255) * 6, (128.0 / 255) * 9, (128.0 / 255) * 9, (128.0 / 255) * 6},
                                        {(128.0 / 255) * 6, (128.0 / 255) * 9, (128.0 / 255) * 9, (128.0 / 255) * 6},
                                        {(128.0 / 255) * 4, (128.0 / 255) * 6, (128.0 / 255) * 6, (128.0 / 255) * 4}},

                                       {{2, 2, 4, 2}, {3, 3, 6, 3}, {3, 3, 6, 3}, {2, 2, 4, 2}},

                                       {{0, 2, 4, 4}, {2, 5, 7, 6}, {4, 7, 8, 6}, {4, 6, 6, 4}},

                                       {{4, 6, 6, 4}, {6, 9, 9, 6}, {6, 9, 9, 6}, {4, 6, 6, 4}},

                                       {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}};
  Tensor<float, 4> ExpectedO(boost::extents[5][4][4][4]);
  Tensor<float, 4> O2(boost::extents[5][4][4][4]);
  for (int n = 0; n < 5; n++) {
    for (int x = 0; x < 4; x++) {
      for (int y = 0; y < 4; y++) {
        for (int c = 0; c < 4; c++) {
          O2[n][x][y][c] = 0;
          for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
              for (int ci = 0; ci < 3; ci++) {
                if (0 <= x + i - 1 && x + i - 1 < 4 && 0 <= y + j - 1 && y + j - 1 < 4) {
                  O2[n][x][y][c] += I[n][x + i - 1][y + j - 1][ci] * K[i][j][ci][c];
                }
              }
            }
          }
        }
      }
    }
  }

  if (VLOG_IS_ON(4)) {
    std::cout << "I: ";
    for (size_t n = 0; n < 5; ++n) {
      for (size_t y = 0; y < 4; ++y) {
        for (size_t x = 0; x < 4; ++x) {
          for (size_t c = 0; c < 3; ++c) {
            std::cout << I[n][y][x][c] << " ";
          }
          std::cout << "\n";
        }
        std::cout << "\n";
      }
      std::cout << "===\n";
    }
  }
  if (VLOG_IS_ON(4)) {
    std::cout << "fc: " << to_string(fc);
    for (size_t l = 0; l < 4; l++) {
      for (size_t i = 0; i < 5; i++) {
        for (size_t k = 0; k < 4; k++) {
          for (size_t j = 0; j < 4; j++) {
            std::cout << O[i][j][k][l];
            std::cout << " ";
          }
          std::cout << "\n";
        }
        std::cout << "\n";
      }
      std::cout << "======\n\n";
    }
  }
  if (VLOG_IS_ON(4)) {
    std::cout << "O2: ";
    for (size_t c = 0; c < 4; ++c) {
      for (size_t n = 0; n < 5; ++n) {
        for (size_t y = 0; y < 4; ++y) {
          for (size_t x = 0; x < 4; ++x) {
            std::cout << std::to_string(O2[n][x][y][c]) << " ";
          }
          std::cout << "\n";
        }
        std::cout << "\n";
      }
      std::cout << "===\n";
    }
    std::cout.flush();
  }

  for (size_t i = 0; i < 5 * 4 * 4 * 4; i++) {
    float tolerance = 0.0001;
    IVLOG(2, "Warning: conv2d test does not handle floats near 0 well!!!")
    IVLOG(3, O.origin()[i] << " vs " << O2.origin()[i]);
    REQUIRE(O.origin()[i] * (1 - tolerance) <= O2.origin()[i]);
    REQUIRE(O.origin()[i] * (1 + tolerance) >= O2.origin()[i]);
  }
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
