#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = smallaios_usb::descriptor::DeviceDescriptor::parse(data);
    let _ = smallaios_usb::descriptor::ConfigDescriptor::parse(data);
    let _ = smallaios_usb::descriptor::InterfaceDescriptor::parse(data);
    let _ = smallaios_usb::descriptor::EndpointDescriptor::parse(data);
    let mut out = [0u8; 256];
    let _ = smallaios_usb::descriptor::parse_string_descriptor(data, &mut out);
});
