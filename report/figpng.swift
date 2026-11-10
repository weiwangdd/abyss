// figpng.swift PDF PAGE SCALE PAD OUT.png
// render one 1-based page of a PDF on white, crop it to the drawn content plus PAD pixels, write a PNG
import Foundation
import PDFKit
import AppKit

let a = CommandLine.arguments
guard a.count == 6, let doc = PDFDocument(url: URL(fileURLWithPath: a[1])), let n = Int(a[2]),
      let page = doc.page(at: n - 1), let scale = Double(a[3]), let pad = Int(a[4]) else {
    print("usage: figpng PDF PAGE SCALE PAD OUT.png")
    exit(1)
}
let box = page.bounds(for: .mediaBox)
let w = Int(box.width * scale), h = Int(box.height * scale)
guard let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: w, pixelsHigh: h, bitsPerSample: 8,
                                 samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                                 colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0),
      let ctx = NSGraphicsContext(bitmapImageRep: rep) else { exit(1) }
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = ctx
ctx.cgContext.setFillColor(NSColor.white.cgColor)
ctx.cgContext.fill(CGRect(x: 0, y: 0, width: w, height: h))
ctx.cgContext.scaleBy(x: CGFloat(scale), y: CGFloat(scale))
page.draw(with: .mediaBox, to: ctx.cgContext)
NSGraphicsContext.restoreGraphicsState()

// bounding box of the pixels that are not white (rows from the top, as the bitmap stores them)
let data = rep.bitmapData!, bpr = rep.bytesPerRow
var x0 = w, y0 = h, x1 = -1, y1 = -1
for y in 0..<h {
    for x in 0..<w {
        let p = data + y * bpr + x * 4
        if p[0] < 245 || p[1] < 245 || p[2] < 245 {
            x0 = min(x0, x); x1 = max(x1, x); y0 = min(y0, y); y1 = max(y1, y)
        }
    }
}
guard x1 >= 0 else { print("blank page"); exit(1) }
x0 = max(0, x0 - pad); y0 = max(0, y0 - pad); x1 = min(w - 1, x1 + pad); y1 = min(h - 1, y1 + pad)
let cw = x1 - x0 + 1, ch = y1 - y0 + 1
guard let cg = rep.cgImage, let cropped = cg.cropping(to: CGRect(x: x0, y: y0, width: cw, height: ch)) else { exit(1) }
let out = NSBitmapImageRep(cgImage: cropped)
try! out.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: a[5]))
print("wrote \(a[5]) \(cw)x\(ch)")
