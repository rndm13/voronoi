#include <fmt/core.h>
#include <fmt/format.h>

#include <cstdio> 
#include <functional>
#include <vector>
#include <cmath>
#include <fstream>
#include <thread>

constexpr size_t WIDTH = 800, HEIGHT = 600;
constexpr size_t SEED_COUNT = 100;

template <typename DataType>
struct Matrix {
    Matrix(size_t _w, size_t _h) : width(_w), height(_h) {
        data = std::vector<DataType>(_w*_h);
    }
    
    DataType& elem(size_t x, size_t y) {
        return data[y * width + x];
    }

    size_t width, height;
    std::vector<DataType> data;
};

struct Pixel {
    int red,green,blue;
    Pixel(int _r,int _g,int _b) : red(_r), green(_g), blue(_b) { }
    Pixel() {}
};

template <> struct fmt::formatter<Pixel> {
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        auto it = ctx.begin(), end = ctx.end();
        if (it != end && *it != '}') throw format_error("invalid format");
        return it;
    }
    template <typename FormatContext>
    auto format(const Pixel& p, FormatContext& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "{} {} {}", p.red, p.green, p.blue);
    }
};


struct Seed {
    size_t x, y;
    Pixel color;
    Seed(size_t _x,size_t _y, Pixel _col) : x(_x), y(_y), color(_col) {}
    Seed() {}
};

double euclidian_distance(Seed a, Seed b) {
    return std::hypot(std::llabs(a.x-b.x),std::llabs(a.y-b.y));
}

double manhattan_distance(Seed a, Seed b) {
    return std::llabs(a.x - b.x) + std::llabs(a.y - b.y);
}

struct Image {
    Matrix<Pixel> data;
    Image(size_t _w,size_t _h) : data(_w,_h) { }
    void render_voronoi(std::vector<Seed> seeds, std::function<double(Seed a,Seed b)> distance_function) {

    #pragma omp parallel for
        for (size_t x = 0; x < data.width;++x) {
            for (size_t y = 0; y < data.height;++y) {
                Seed cur(x, y, Pixel(0,0,0));
                data.elem(x,y) = std::min_element(seeds.begin(),seeds.end(),
                        [&](Seed a,Seed b){
                            return distance_function(a,cur) <= distance_function(b,cur);
                        }) -> color;
            }
        }
    }
    void write_ppm(const std::string& filename) {
        std::ofstream outstream(filename);
        outstream << fmt::format("P3\n{} {}\n255\n",data.width,data.height);
        outstream << fmt::format("{}",fmt::join(data.data," "));
    }
};

typedef std::function<Pixel(size_t,size_t)> color_gen;

Pixel lerp_pixels(Pixel a, Pixel b,double t) {
    return Pixel( std::lerp(a.red,b.red,t)
                , std::lerp(a.green,b.green,t)
                , std::lerp(a.blue,b.blue,t));
}

auto rand_color = [](size_t, size_t){
    return Pixel(rand()%256,rand()%256,rand()%256);
};

color_gen make_cg_horizontal_gradient(Pixel begin_color, Pixel end_color) {
    return [=](size_t x, size_t) {
        return lerp_pixels(begin_color,end_color,x/(static_cast<double>(WIDTH)));
    };
}

color_gen make_cg_vertical_gradient(Pixel begin_color, Pixel end_color) {
    return [=](size_t, size_t y) {
        return lerp_pixels(begin_color,end_color,y/(static_cast<double>(HEIGHT)));
    };
}

std::vector<Seed> gen_random_seeds(size_t count, size_t width, size_t height, color_gen cg) {
    std::vector<Seed> result(count);

    std::generate(result.begin(),result.end(),
            [&]() mutable {
                size_t x = rand()%width;
                size_t y = rand()%height;
                auto col = cg(x,y);
                // fmt::print("({},{}) = {}\n",x,y,col);
                return Seed(x,y,col);
            });

    return result;
}

int main() {
    srand(time(NULL));

    auto seeds = gen_random_seeds(SEED_COUNT,WIDTH,HEIGHT,make_cg_vertical_gradient(Pixel(0,0,0),Pixel(255,255,255)));

    std::thread thr1 ([&] {
        Image eucl(WIDTH, HEIGHT);
        eucl.render_voronoi(seeds,euclidian_distance);
        eucl.write_ppm("output_euclidian.ppm");
    });

    std::thread thr2([&] {
        Image manh(WIDTH, HEIGHT);
        manh.render_voronoi(seeds,manhattan_distance);
        manh.write_ppm("output_manhattan.ppm");
    });

    thr1.join();
    thr2.join();

    // std::system("gimp output_*");
    return EXIT_SUCCESS;
}
