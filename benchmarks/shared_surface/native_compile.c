// Compile-only diagnostic for a captured software-tracing compute shader.
// Usage: native_compile DEVICE_SUBSTRING SHADER.spv [no-opt]
// No dispatch is submitted. Use an external time and data-memory limit.
#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VK_CHECK(expr) do { VkResult r = (expr); if (r != VK_SUCCESS) { \
    fprintf(stderr, "%s: VkResult %d\n", #expr, r); exit(2); } } while (0)

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s DEVICE_SUBSTRING SHADER.spv [no-opt]\n", argv[0]);
        return 2;
    }
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "material native compile", .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app};
    VkInstance instance;
    VK_CHECK(vkCreateInstance(&ici, NULL, &instance));
    uint32_t count;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, NULL));
    VkPhysicalDevice *devices = calloc(count, sizeof(*devices));
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, devices));
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < count; ++i) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        if (strstr(props.deviceName, argv[1])) {
            physical = devices[i];
            printf("Device: %s\n", props.deviceName);
            break;
        }
    }
    free(devices);
    if (!physical) { fprintf(stderr, "Device not found\n"); return 2; }
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, NULL);
    VkQueueFamilyProperties *families = calloc(count, sizeof(*families));
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families);
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < count; ++i) {
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { family = i; break; }
    }
    free(families);
    if (family == UINT32_MAX) { fprintf(stderr, "No compute queue\n"); return 2; }
    float priority = 1;
    VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &priority};
    VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR workgroup = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR,
        .workgroupMemoryExplicitLayout = VK_TRUE};
    VkPhysicalDeviceVulkan13Features f13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .pNext = &workgroup,
        .maintenance4 = VK_TRUE, .shaderZeroInitializeWorkgroupMemory = VK_TRUE};
    VkPhysicalDeviceVulkan12Features f12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &f13, .bufferDeviceAddress = VK_TRUE, .scalarBlockLayout = VK_TRUE,
        .vulkanMemoryModel = VK_TRUE, .vulkanMemoryModelDeviceScope = VK_TRUE};
    VkPhysicalDeviceVulkan11Features f11 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, .pNext = &f12,
        .variablePointers = VK_TRUE, .variablePointersStorageBuffer = VK_TRUE};
    VkPhysicalDeviceFeatures features = {.shaderInt64 = VK_TRUE};
    const char *extensions[] = {VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME};
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &f11, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
        .enabledExtensionCount = 1, .ppEnabledExtensionNames = extensions,
        .pEnabledFeatures = &features};
    VkDevice device;
    VK_CHECK(vkCreateDevice(physical, &dci, NULL, &device));
    VkQueue queue;
    vkGetDeviceQueue(device, family, 0, &queue);

    FILE *file = fopen(argv[2], "rb");
    if (!file) { perror(argv[2]); return 2; }
    fseek(file, 0, SEEK_END);
    long code_size = ftell(file);
    rewind(file);
    uint32_t *code = malloc((size_t)code_size);
    if (fread(code, 1, (size_t)code_size, file) != (size_t)code_size) { fprintf(stderr, "Short shader read\n"); return 2; }
    fclose(file);
    VkShaderModuleCreateInfo smci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = (size_t)code_size, .pCode = code};
    VkShaderModule shader;
    VK_CHECK(vkCreateShaderModule(device, &smci, NULL, &shader));
    free(code);
    VkPushConstantRange range = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = sizeof(VkDeviceAddress)};
    VkPipelineLayoutCreateInfo plci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &range};
    VkPipelineLayout layout;
    VK_CHECK(vkCreatePipelineLayout(device, &plci, NULL, &layout));
    VkComputePipelineCreateInfo cpci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main"},
        .layout = layout};
    if (argc > 3 && strcmp(argv[3], "no-opt") == 0) cpci.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
    puts("PIPELINE_COMPILE_START"); fflush(stdout);
    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &pipeline));
    puts("PIPELINE_COMPILE_DONE");
    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyPipelineLayout(device, layout, NULL);
    vkDestroyShaderModule(device, shader, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    return 0;
}
