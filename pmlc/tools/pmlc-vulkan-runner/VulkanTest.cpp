#include <vulkan/vulkan.h>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>

class HelloTriangleApplication {
 public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

 private:
  void initWindow() {}

  void initVulkan() { createInstance(); }

  void mainLoop() {}

  void cleanup() {}

  void createInstance() {
    VkApplicationInfo applicationInfo = {};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    // applicationInfo.pNext = nullptr;
    applicationInfo.pApplicationName = "MLIR Vulkan runtime";
    applicationInfo.applicationVersion = 0;
    applicationInfo.pEngineName = "mlir";
    applicationInfo.engineVersion = 0;
    applicationInfo.apiVersion = VK_MAKE_VERSION(1, 0, 0);

    VkInstanceCreateInfo instanceCreateInfo = {};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    // instanceCreateInfo.pNext = nullptr;
    instanceCreateInfo.flags = 0;
    instanceCreateInfo.pApplicationInfo = &applicationInfo;
    instanceCreateInfo.enabledLayerCount = 0;
    instanceCreateInfo.ppEnabledLayerNames = 0;
    instanceCreateInfo.enabledExtensionCount = 0;
    instanceCreateInfo.ppEnabledExtensionNames = 0;

    VkResult result = vkCreateInstance(&instanceCreateInfo, 0, &instance);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("failed to create instance!");
    } else {
      std::cout << "Result=VK_SUCCESS" << std::endl;
    }
  }

 private:
  VkInstance instance;
};

int main() {
  HelloTriangleApplication app;

  try {
    app.run();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
