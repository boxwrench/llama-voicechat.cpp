set(TARGET llama-voicechat)
add_executable(${TARGET} voicechat-cli.cpp voicechat-tts.cpp)
target_link_libraries(${TARGET} PRIVATE llama llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../mtmd)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
