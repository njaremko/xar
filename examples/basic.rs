use xar::Xar;

fn main() {
    let mut values = Xar::new();

    let first = values.push_ptr(String::from("first"));
    for i in 0..1_000 {
        values.push(i.to_string());
    }

    println!("len = {}, capacity = {}", values.len(), values.capacity());
    println!("first = {}", unsafe { first.as_ref() });

    for chunk in values.chunks() {
        println!("chunk length = {}", chunk.len());
    }
}
