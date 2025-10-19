#include <nanobind/nanobind.h>
#include <vector>

#define PTX_INJECT_IMPLEMENTATION
#include <ptx_inject.h>

namespace nb = nanobind;

NB_MODULE(_ptx_inject, m) {
    nb::enum_<PtxInjectResult>(m, "PtxInjectResult", "PTX Inject status type returns")
        .value("PTX_INJECT_SUCCESS",                            PTX_INJECT_SUCCESS,                             "PTX Inject Operation was successful")
        .value("PTX_INJECT_ERROR_FORMATTING",                   PTX_INJECT_ERROR_FORMATTING,                    "PTX Inject formatting is wrong.")
        .value("PTX_INJECT_ERROR_INSUFFICIENT_BUFFER",          PTX_INJECT_ERROR_INSUFFICIENT_BUFFER,           "The buffer passed in is not large enough.")
        .value("PTX_INJECT_ERROR_INTERNAL",                     PTX_INJECT_ERROR_INTERNAL,                      "An internal error occurred.")
        .value("PTX_INJECT_ERROR_INVALID_INPUT",                PTX_INJECT_ERROR_INVALID_INPUT,                 "An value passed to the function is wrong.")
        .value("PTX_INJECT_ERROR_MAX_UNIQUE_INJECTS_EXCEEDED",  PTX_INJECT_ERROR_MAX_UNIQUE_INJECTS_EXCEEDED,   "The amount of injects found in the file exceeds the maximum.")
        .value("PTX_INJECT_ERROR_WRONG_NUM_STUBS",              PTX_INJECT_ERROR_WRONG_NUM_STUBS,               "The amount of stubs passed in does not match the amount of injects found in the file.")
        .value("PTX_INJECT_ERROR_OUT_OF_BOUNDS_IDX",            PTX_INJECT_ERROR_OUT_OF_BOUNDS_IDX,             "The index passed in is out of bounds of the range of values being indexed.")
        .value("PTX_INJECT_ERROR_INCONSISTENT_INJECTION",       PTX_INJECT_ERROR_INCONSISTENT_INJECTION,        "An inject site found in the file has a different signature than another inject site found with the same name.")
        .value("PTX_INJECT_ERROR_INJECTION_NAME_NOT_FOUND",     PTX_INJECT_ERROR_INJECTION_NAME_NOT_FOUND,      "Inject name not found.")
        .value("PTX_INJECT_ERROR_INJECTION_ARG_NAME_NOT_FOUND", PTX_INJECT_ERROR_INJECTION_ARG_NAME_NOT_FOUND,  "Inject arg name not found.")
        .value("PTX_INJECT_ERROR_OUT_OF_MEMORY",                PTX_INJECT_ERROR_OUT_OF_MEMORY,                 "PTX Inject is out of memory, malloc failed.")
        .value("PTX_INJECT_RESULT_NUM_ENUMS",                   PTX_INJECT_RESULT_NUM_ENUMS,                    "The number of result enums.")
        .export_values();

    nb::enum_<PtxInjectMutType>(m, "PtxInjectMutType", "Specifies how the inline-ptx sets the variable, as modifiable, output or input.")
        .value("PTX_INJECT_MUT_TYPE_OUT",       PTX_INJECT_MUT_TYPE_OUT)
        .value("PTX_INJECT_MUT_TYPE_MOD",       PTX_INJECT_MUT_TYPE_MOD)
        .value("PTX_INJECT_MUT_TYPE_IN",        PTX_INJECT_MUT_TYPE_IN)
        .value("PTX_INJECT_MUT_TYPE_NUM_ENUMS", PTX_INJECT_MUT_TYPE_NUM_ENUMS)
        .export_values();

    m.def(
        "ptx_inject_result_to_string", 
        &ptx_inject_result_to_string, 
        nb::arg("PtxInjectResult"),
        "Converts a PtxInjectResult enum value to a human-readable string."
    );

    m.def(
        "ptx_inject_process_cuda",
        [](
            nb::list data_type_info_names,
            nb::list data_type_info_register_types,
            nb::list data_type_info_mov_postfixes,
            nb::list data_type_info_register_char,
            nb::list data_type_info_register_cast_str,
            const char* annotated_cuda_src, 
            nb::object processed_cuda_buffer_obj
        ) {
            size_t num_data_type_infos = data_type_info_names.size();
            if (
                (size_t) data_type_info_names.size()               != num_data_type_infos ||
                (size_t) data_type_info_mov_postfixes.size()       != num_data_type_infos ||
                (size_t) data_type_info_register_char.size()       != num_data_type_infos ||
                (size_t) data_type_info_register_cast_str.size()   != num_data_type_infos
            ) {
                nb::raise_type_error("All data_type_info lists must have identical length == num_data_type_infos");
            }

            size_t processed_cuda_buffer_size = 0;
            size_t processed_cuda_bytes_written = 0;
            size_t num_inject_sites = 0;

            char* processed_cuda_buffer = nullptr;

            std::vector<PtxInjectDataTypeInfo> data_type_infos(num_data_type_infos);
            for (size_t i = 0; i < num_data_type_infos; i++) {
                PtxInjectDataTypeInfo data_type_info = {
                    nb::cast<const char*>(data_type_info_names[i]),
                    nb::cast<const char*>(data_type_info_register_types[i]),
                    nb::cast<const char*>(data_type_info_mov_postfixes[i]),
                    nb::cast<char>(data_type_info_register_char[i]),
                    nb::cast<const char*>(data_type_info_register_cast_str[i])
                };
                data_type_infos[i] = data_type_info;
            }

            if (nb::isinstance<nb::bytearray>(processed_cuda_buffer_obj)) {
                nb::bytearray ba = nb::cast<nb::bytearray>(processed_cuda_buffer_obj);
                processed_cuda_buffer = reinterpret_cast<char*>(ba.data());
                processed_cuda_buffer_size = ba.size();
            } else if (!processed_cuda_buffer_obj.is_none()) {
                nb::raise_type_error("processed_cuda_buffer must by bytearray or None");
            }

            PtxInjectResult result = ptx_inject_process_cuda(
                data_type_infos.data(),
                num_data_type_infos,
                annotated_cuda_src,
                processed_cuda_buffer,
                processed_cuda_buffer_size,
                &processed_cuda_bytes_written,
                &num_inject_sites
            );
                
            return nb::make_tuple(result, processed_cuda_bytes_written, num_inject_sites);
        },
        nb::arg("data_type_info_names"),
        nb::arg("data_type_info_register_types"),
        nb::arg("data_type_info_mov_postfixes"),
        nb::arg("data_type_info_register_char"),
        nb::arg("data_type_info_register_cast_str"),
        nb::arg("annotated_cuda_src"),
        nb::arg("buffer").none(),
        "Process CUDA source code with PTX injection, returning the result, processed buffer, bytes written, and number of injection sites."
    );

    m.def(
        "ptx_inject_create", 
        [](
            nb::list data_type_info_names,
            nb::list data_type_info_register_types,
            nb::list data_type_info_mov_postfixes,
            nb::list data_type_info_register_char,
            nb::list data_type_info_register_cast_str,
            const char* processed_ptx_src
        ) {
            size_t num_data_type_infos = data_type_info_names.size();
            if (
                (size_t) data_type_info_names.size()               != num_data_type_infos ||
                (size_t) data_type_info_mov_postfixes.size()       != num_data_type_infos ||
                (size_t) data_type_info_register_char.size()       != num_data_type_infos ||
                (size_t) data_type_info_register_cast_str.size()   != num_data_type_infos
            ) {
                nb::raise_type_error("All data_type_info lists must have identical length == num_data_type_infos");
            }

            std::vector<PtxInjectDataTypeInfo> data_type_infos(num_data_type_infos);
            for (size_t i = 0; i < num_data_type_infos; i++) {
                PtxInjectDataTypeInfo data_type_info = {
                    nb::cast<const char*>(data_type_info_names[i]),
                    nb::cast<const char*>(data_type_info_register_types[i]),
                    nb::cast<const char*>(data_type_info_mov_postfixes[i]),
                    nb::cast<char>(data_type_info_register_char[i]),
                    nb::cast<const char*>(data_type_info_register_cast_str[i])
                };
                data_type_infos[i] = data_type_info;
            }

            PtxInjectHandle handle = nullptr;
            PtxInjectResult result = 
                ptx_inject_create(
                    &handle, 
                    data_type_infos.data(),
                    num_data_type_infos,
                    processed_ptx_src
                );
            
            return nb::make_tuple(
                nb::cast(result),
                handle ? nb::capsule(handle, "PtxInjectHandle") : nb::none()
            );
        },
        nb::arg("data_type_info_names"),
        nb::arg("data_type_info_register_types"),
        nb::arg("data_type_info_mov_postfixes"),
        nb::arg("data_type_info_register_char"),
        nb::arg("data_type_info_register_cast_str"),
        nb::arg("processed_ptx_src"),
       "Creates a PtxInject handle from PTX source. Returns a tuple (result, handle)."
    );

    m.def(
        "ptx_inject_destroy", 
        [](nb::capsule handle) {
            PtxInjectHandle ptx_handle = static_cast<PtxInjectHandle>(handle.data());
            PtxInjectResult result = ptx_inject_destroy(ptx_handle);
            return nb::cast(result);
        }, 
        nb::arg("handle"),
       "Destroys a PtxInject handle. Returns the result of the operation."
    );

    m.def(
        "ptx_inject_num_injects", 
        [](
            nb::capsule handle
        ) {
            PtxInjectHandle ptx_handle = static_cast<PtxInjectHandle>(handle.data());
            size_t num_injects = 0;
            PtxInjectResult result = ptx_inject_num_injects(ptx_handle, &num_injects);
            
            return nb::make_tuple(
                nb::cast(result),
                nb::cast(num_injects)
            );
        }, 
        nb::arg("handle"),
       "Gets the number of injections for a PtxInject handle. Returns a tuple (result, num_injects)."
    );

    m.def(
        "ptx_inject_inject_info_by_name", 
        [](
            nb::capsule handle, 
            const char* inject_name
        ) {
            PtxInjectHandle ptx_handle = static_cast<PtxInjectHandle>(handle.data());
            size_t inject_idx = 0;
            size_t inject_num_args = 0;
            size_t inject_num_sites = 0;
            PtxInjectResult result = 
                ptx_inject_inject_info_by_name(
                    ptx_handle, 
                    inject_name, 
                    &inject_idx, 
                    &inject_num_args, 
                    &inject_num_sites
                );
            
            return nb::make_tuple(
                nb::cast(result),
                nb::cast(inject_idx),
                nb::cast(inject_num_args),
                nb::cast(inject_num_sites)
            );
        }, 
        nb::arg("handle"), 
        nb::arg("inject_name"),
       "Gets injection info by name for a PtxInject handle. Returns a tuple (result, inject_idx, inject_num_args, inject_num_sites)."
    );

    m.def(
        "ptx_inject_inject_info_by_index", 
        [](
            nb::capsule handle, 
            size_t inject_idx
        ) {
            PtxInjectHandle ptx_handle = static_cast<PtxInjectHandle>(handle.data());
            const char* inject_name = nullptr;
            size_t inject_num_args = 0;
            size_t inject_num_sites = 0;
            PtxInjectResult result = 
                ptx_inject_inject_info_by_index(
                    ptx_handle, 
                    inject_idx, 
                    &inject_name, 
                    &inject_num_args,
                    &inject_num_sites
                );
            
            return nb::make_tuple(
                nb::cast(result),
                inject_name ? nb::cast(inject_name) : nb::none(),
                nb::cast(inject_num_args),
                nb::cast(inject_num_sites)
            );
        }, 
        nb::arg("handle"), 
        nb::arg("inject_idx"),
       "Gets injection info by index for a PtxInject handle. Returns a tuple (result, inject_name, inject_num_args, inject_num_sites)."
    );

    m.def(
        "ptx_inject_variable_info_by_name", 
        [](
            nb::capsule handle, 
            size_t inject_idx, 
            const char* inject_variable_name
        ) {
            PtxInjectHandle ptx_handle = static_cast<PtxInjectHandle>(handle.data());
            size_t inject_variable_arg_idx = 0; // Initialize output parameters
            PtxInjectMutType inject_variable_mut_type;
            size_t inject_variable_data_type;
            const char* inject_variable_stable_register_name = nullptr;
            PtxInjectResult result = 
                ptx_inject_variable_info_by_name(
                    ptx_handle, 
                    inject_idx, 
                    inject_variable_name, 
                    &inject_variable_arg_idx, 
                    &inject_variable_mut_type, 
                    &inject_variable_data_type, 
                    &inject_variable_stable_register_name
                );
            
            return nb::make_tuple(
                nb::cast(result),
                nb::cast(inject_variable_arg_idx),
                nb::cast(inject_variable_mut_type),
                nb::cast(inject_variable_data_type),
                inject_variable_stable_register_name ? nb::cast(inject_variable_stable_register_name) : nb::none()
            );
        }, 
        nb::arg("handle"), 
        nb::arg("inject_idx"), 
        nb::arg("inject_variable_name"),
       "Gets variable info by name for a PtxInject handle. Returns a tuple (result, inject_variable_arg_idx, inject_variable_mut_type, inject_variable_data_type, inject_variable_stable_register_name)."
    );

    m.def(
        "ptx_inject_variable_info_by_index", 
        [](
            nb::capsule handle, 
            size_t inject_idx, 
            size_t inject_variable_arg_idx
        ) {
            PtxInjectHandle ptx_handle = static_cast<PtxInjectHandle>(handle.data());
            const char* inject_variable_name = nullptr; // Initialize output parameters
            PtxInjectMutType inject_variable_mut_type;
            size_t inject_variable_data_type;
            const char* inject_variable_stable_register_name = nullptr;
            PtxInjectResult result = 
                ptx_inject_variable_info_by_index(
                    ptx_handle, 
                    inject_idx, 
                    inject_variable_arg_idx, 
                    &inject_variable_name, 
                    &inject_variable_mut_type, 
                    &inject_variable_data_type, 
                    &inject_variable_stable_register_name
                );
            
            return nb::make_tuple(
                nb::cast(result),
                inject_variable_name ? nb::cast(inject_variable_name) : nb::none(),
                nb::cast(inject_variable_mut_type),
                nb::cast(inject_variable_data_type),
                inject_variable_stable_register_name ? nb::cast(inject_variable_stable_register_name) : nb::none()
            );
        }, 
        nb::arg("handle"), 
        nb::arg("inject_idx"), 
        nb::arg("inject_variable_arg_idx"),
       "Gets variable info by index for a PtxInject handle. Returns a tuple (result, inject_variable_name, inject_variable_mut_type, inject_variable_data_type, inject_variable_stable_register_name)."
    );

    m.def(
        "ptx_inject_render_ptx", 
        [](
            nb::capsule handle, 
            nb::list ptx_stubs, 
            nb::object rendered_ptx_buffer_obj
        ) {
            PtxInjectHandle ptx_handle = static_cast<PtxInjectHandle>(handle.data());
            
            size_t num_ptx_stubs = ptx_stubs.size();
            std::vector<const char*> ptx_stubs_array(num_ptx_stubs);
            for (size_t i = 0; i < num_ptx_stubs; ++i) {
                ptx_stubs_array[i] = nb::cast<const char*>(ptx_stubs[i]);
            }
            
            size_t rendered_ptx_buffer_size = 0;
            size_t rendered_ptx_bytes_written = 0;
            char* rendered_ptx_buffer = nullptr;
            if (nb::isinstance<nb::bytearray>(rendered_ptx_buffer_obj)) {
                nb::bytearray ba = nb::cast<nb::bytearray>(rendered_ptx_buffer_obj);
                rendered_ptx_buffer = reinterpret_cast<char*>(ba.data());
                rendered_ptx_buffer_size = ba.size();
            } else if (!rendered_ptx_buffer_obj.is_none()) {
                nb::raise_type_error("rendered_ptx_buffer must be bytearray or None");
            }
            
            PtxInjectResult result = ptx_inject_render_ptx(
                ptx_handle,
                num_ptx_stubs > 0 ? ptx_stubs_array.data() : nullptr,
                num_ptx_stubs,
                rendered_ptx_buffer,
                rendered_ptx_buffer_size,
                &rendered_ptx_bytes_written
            );
            
            return nb::make_tuple(
                nb::cast(result),
                nb::cast(rendered_ptx_bytes_written)
            );
        }, 
        nb::arg("handle"), 
        nb::arg("ptx_stubs"), 
        nb::arg("rendered_ptx_buffer") = nb::none(),
       "Renders PTX code for a PtxInject handle. Returns a tuple (result, rendered_ptx_bytes_written)."
    );
}