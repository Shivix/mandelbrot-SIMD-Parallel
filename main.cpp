#include <iostream>
#include <opencv2/opencv.hpp>
#include "vectorclass.h"
#include <thread>

#if defined(__AVX2__) && defined(__linux__)
#include <immintrin.h>

struct ComplexVec4d{
    Vec4d real;
    Vec4d imag;
    
    ComplexVec4d operator+ (const ComplexVec4d rhs){
        real += rhs.real;
        imag += rhs.imag;
        return *this;
    }
};
ComplexVec4d mandelbrot(const ComplexVec4d z, const ComplexVec4d c){
    auto zr{z.real * z.real - z.imag * z.imag + c.real};
    auto zi{z.real * z.imag * 2.0 + c.imag};
    
    return {zr, zi};
}

Vec4db escape_check(const ComplexVec4d z){
    return (z.real * z.real + z.imag * z.imag) > 4;
}

Vec4i compute_point(const ComplexVec4d c){ // TODO: improve
    constexpr int MAX_ITER{255 * 4};
    ComplexVec4d z{0, 0};
    Vec4i result{MAX_ITER, MAX_ITER, MAX_ITER, MAX_ITER};
    Vec4ib r{false, false, false, false};
    for(int i{1}; i <= MAX_ITER; ++i){
        z = mandelbrot(z, c);
        auto iters{escape_check(z)};
        if (iters[0] && !r[0]){
            result.insert(0, i);
            r.insert(0, true);
        }
        if (iters[1] && !r[1]){
            result.insert(1, i);
            r.insert(1, true);
        }
        if (iters[2] && !r[2]){
            result.insert(2, i);
            r.insert(2, true);
        }
        if (iters[3] && !r[3]){
            result.insert(3, i);
            r.insert(3, true);
        }
        if(horizontal_and(r)){
            return result;
        }
    }
    return result;
}

Vec4d pixel_to_mandelbrot(const Vec4d pixel, const double offset, const double zoom, const int zoom_correction){
    return (pixel - zoom_correction) * zoom + offset;
}

// Utilizes AVX2 for SIMD  --  256 bit registers with 4 doubles
void render_mandelbrot(cv::Mat& mandelbrot_set, const double x_offset, const double y_offset, const double zoom,
                       const int width, const int height, const int width_start = 0, const int height_start = 0){
    auto width_correction{mandelbrot_set.cols / 2};
    auto height_correction{mandelbrot_set.rows / 2};
    for(auto i{height_start}; i < height; ++i){
        for(auto j{width_start}; j < width; j+=4){
            Vec4d pixels{static_cast<double>(j), static_cast<double>(j+1), static_cast<double>(j+2), static_cast<double>(j+3)};
            const Vec4i num_of_iters{compute_point({pixel_to_mandelbrot(pixels, x_offset, zoom, width_correction),
                                                   pixel_to_mandelbrot(i, y_offset, zoom, height_correction)})};
            
            const auto colour{static_cast<unsigned char>(num_of_iters[0] % 255)};
            mandelbrot_set.at<cv::Vec3b>(i, j) = cv::Vec3b{colour, colour, colour};
            const auto colour2{static_cast<unsigned char>(num_of_iters[1] % 255)};
            mandelbrot_set.at<cv::Vec3b>(i, j + 1) = cv::Vec3b{colour2, colour2, colour2};
            const auto colour3{static_cast<unsigned char>(num_of_iters[2] % 255)};
            mandelbrot_set.at<cv::Vec3b>(i, j + 2) = cv::Vec3b{colour3, colour3, colour3};
            const auto colour4{static_cast<unsigned char>(num_of_iters[3] % 255)};
            mandelbrot_set.at<cv::Vec3b>(i, j + 3) = cv::Vec3b{colour4, colour4, colour4};
        }
    }
}

#else

#include <complex>
// z = z^2 + point to assess
std::complex<double> mandelbrot(std::complex<double> z, std::complex<double> c){
    return z * z + c;
}

bool escape_check(std::complex<double> z){
    return (z.real() * z.real() + z.imag() * z.imag()) > 4;
}

int compute_point(const std::complex<double> c){
    constexpr int MAX_ITER{255 * 3}; // maximum amount of iterations before considering it "infinite"
    std::complex<double> z{0, 0};
    for(int i{1}; i <= MAX_ITER; ++i){
        z = mandelbrot(z, c);
        if (escape_check(z)){
            return i;
        }
    }
    return MAX_ITER;
}

double pixel_to_mandelbrot(const int pixel, const double offset, const double zoom, const int zoom_correction){
    return (pixel - zoom_correction) * zoom + offset;
}

void render_mandelbrot(cv::Mat& mandelbrot_set, const double x_offset, const double y_offset, const double zoom,
                       const size_t width, const size_t height, const size_t width_start = 0, const size_t height_start = 0){
    auto width_correction{width / 2};
    auto height_correction{height / 2};
    for(auto i{width_start}; i < mandelbrot_set.size().height; ++i){
        for(auto j{height_start}; j < mandelbrot_set.size().width; ++j){
            const int iterations{compute_point(std::complex<double>{pixel_to_mandelbrot(i, x_offset, zoom, width_correction), 
                                                                    pixel_to_mandelbrot(j, y_offset, zoom, height_correction)})};
            
            const auto colour{static_cast<unsigned char>(iterations % 255)};
            mandelbrot_set.at<cv::Vec3b>(j, i) = cv::Vec3b{colour, colour, colour};
        }
    }
}

#endif

void render_mandelbrot_threaded(std::array<std::thread, 4>& thread_pool, cv::Mat& mandelbrot_set,
                                const double x_offset, const double y_offset, const double zoom, const int width, const int height){
    auto width_divider{width / 4}; // TODO: use max threads available
    thread_pool[0] = std::thread{[&](){
        render_mandelbrot(mandelbrot_set, x_offset, y_offset, zoom, width_divider, height);
    }};
    thread_pool[1] = std::thread{[&](){
        render_mandelbrot(mandelbrot_set, x_offset, y_offset, zoom, width_divider * 2, height, width_divider);
    }};
    thread_pool[2] = std::thread{[&](){
        render_mandelbrot(mandelbrot_set, x_offset, y_offset, zoom, width_divider * 3, height, width_divider * 2);
    }};
    thread_pool[3] = std::thread{[&](){
        render_mandelbrot(mandelbrot_set, x_offset, y_offset, zoom, width, height, width_divider * 3);
    }};
    for(auto&& thread: thread_pool){
        thread.join();
    }
}

int main(){
    constexpr int WIDTH{1920};
    constexpr int HEIGHT{1080};
    
    cv::Mat mandelbrot_set{cv::Size(WIDTH, HEIGHT), CV_8UC3, cv::Scalar(30, 30, 30)};
    std::array<std::thread, 4> thread_pool;
    
    double zoom{0.003};
    double x_offset{0.0};
    double y_offset{0.0};
    bool is_running{true};
    while (is_running){
        auto start_point{std::chrono::high_resolution_clock::now()};
        render_mandelbrot_threaded(thread_pool, mandelbrot_set, x_offset, y_offset, zoom, WIDTH, HEIGHT);
        cv::imshow("Mandelbrot Set", mandelbrot_set);
        auto end_point{std::chrono::high_resolution_clock::now()};
        std::chrono::duration<double> elapsed_time{end_point - start_point};
        std::cout << elapsed_time.count() << std::endl;
        auto key_pressed{cv::waitKey(0)};
        switch(key_pressed){
            case 'w':
                y_offset -= zoom * 40.0;
                break;
            case 'a':
                x_offset -= zoom * 40.0;
                break;
            case 's':
                y_offset += zoom * 40.0;
                break;
            case 'd':
                x_offset += zoom * 40.0;
                break;
            case 'q':
                zoom *= -0.8;
                break;
            case 'e':
                zoom *= 0.8;
                break;
            default:
                is_running = false;
                break;
        }
    }
    return 0;
}