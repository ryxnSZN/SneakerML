/*
See LICENSE folder for this sample’s licensing information.

Abstract:
View controller for selecting images and applying Vision + Core ML processing.
*/

import UIKit
import CoreML
import Vision
import ImageIO

enum Sneaker: String {
    case dunkSB = "dunk_sb"
    case yeezyBoost350 = "yeezy_boost_350"
    case ultraboost = "ultraboost"
    case airForce1 = "air_force_1"
    case airPresto = "air_presto"
    case airJordan1 = "air_jordan_1"
    case airJordan3 = "air_jordan_3"
    case airJordan4 = "air_jordan_4"
    case airJordan5 = "air_jordan_5"
    case airJordan6 = "air_jordan_6"
    case airJordan7 = "air_jordan_7"
}

class ImageClassificationViewController: UIViewController {
    // MARK: - IBOutlets
    
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var cameraButton: UIBarButtonItem!
    @IBOutlet weak var classificationLabel: UILabel!
    
    // MARK: - Image Classification
    
    /// - Tag: MLModelSetup
    lazy var classificationRequest: VNCoreMLRequest = {
        do {
            /*
             Use the Swift class `MobileNet` Core ML generates from the model.
             To use a different Core ML classifier model, add it to the project
             and replace `MobileNet` with that model's generated Swift class.
             */
            let model = try VNCoreMLModel(for: AppModel().model)
            
            let request = VNCoreMLRequest(model: model, completionHandler: { [weak self] request, error in
                self?.processClassifications(for: request, error: error)
            })
            request.imageCropAndScaleOption = .centerCrop
            return request
        } catch {
            fatalError("Failed to load Vision ML model: \(error)")
        }
    }()
    
    /// - Tag: PerformRequests
    func updateClassifications(for image: UIImage) {
        classificationLabel.text = "Classifying..."
        
        let orientation = CGImagePropertyOrientation(image.imageOrientation)
        guard let ciImage = CIImage(image: image) else { fatalError("Unable to create \(CIImage.self) from \(image).") }
        
        DispatchQueue.global(qos: .userInitiated).async {
            let handler = VNImageRequestHandler(ciImage: ciImage, orientation: orientation)
            do {
                try handler.perform([self.classificationRequest])
            } catch {
                /*
                 This handler catches general image processing errors. The `classificationRequest`'s
                 completion handler `processClassifications(_:error:)` catches errors specific
                 to processing that request.
                 */
                print("Failed to perform classification.\n\(error.localizedDescription)")
            }
        }
    }
    
    /// Updates the UI with the results of the classification.
    /// - Tag: ProcessClassifications
    func processClassifications(for request: VNRequest, error: Error?) {
        DispatchQueue.main.async {
            guard let results = request.results else {
                self.classificationLabel.text = "Unable to classify image.\n\(error!.localizedDescription)"
                return
            }
            // The `results` will always be `VNClassificationObservation`s, as specified by the Core ML model in this project.
            let classifications = results as! [VNClassificationObservation]
        
            if classifications.isEmpty {
                self.classificationLabel.text = "Nothing recognized."
            } else {
                // Display top classifications ranked by confidence in the UI.
                let topClassifications = classifications.prefix(2).filter { $0.confidence >= 0.0001 }
                let descriptions = topClassifications.map { classification -> String in
                    // Formats the classification for display; e.g. "(0.37) cliff, drop, drop-off".
                    switch classification.identifier {
                        case Sneaker.dunkSB.rawValue:
                            return String(format: "Nike Dunk SB - %.2f%%", classification.confidence * 100)
                        case Sneaker.yeezyBoost350.rawValue:
                            return String(format: "Adidas Yeezy Boost 350 - %.2f%%", classification.confidence * 100)
                        case Sneaker.ultraboost.rawValue:
                            return String(format: "Adidas Ultraboost - %.2f%%", classification.confidence * 100)
                        case Sneaker.airForce1.rawValue:
                            return String(format: "Nike Air Force 1 - %.2f%%", classification.confidence * 100)
                        case Sneaker.airPresto.rawValue:
                            return String(format: "Nike Air Presto - %.2f%%", classification.confidence * 100)
                        case Sneaker.airJordan1.rawValue:
                            return String(format: "Nike Air Jordan 1 - %.2f%%", classification.confidence * 100)
                        case Sneaker.airJordan3.rawValue:
                            return String(format: "Nike Air Jordan 3 - %.2f%%", classification.confidence * 100)
                        case Sneaker.airJordan4.rawValue:
                            return String(format: "Nike Air Jordan 4 - %.2f%%", classification.confidence * 100)
                        case Sneaker.airJordan5.rawValue:
                            return String(format: "Nike Air Jordan 5 - %.2f%%", classification.confidence * 100)
                        case Sneaker.airJordan6.rawValue:
                            return String(format: "Nike Air Jordan 6 - %.2f%%", classification.confidence * 100)
                        case Sneaker.airJordan7.rawValue:
                            return String(format: "Nike Air Jordan 7 - %.2f%%", classification.confidence * 100)
                        default:
                            break
                    }
                    return String(format: "%@ - %.2f%%", classification.identifier, classification.confidence * 100)
                }
                self.classificationLabel.text = "Prediction:\n" + descriptions.joined(separator: "\n")
            }
        }
    }
    
    // MARK: - Photo Actions
    
    @IBAction func takePicture() {
        // Show options for the source picker only if the camera is available.
        guard UIImagePickerController.isSourceTypeAvailable(.camera) else {
            presentPhotoPicker(sourceType: .photoLibrary)
            return
        }
        
        let photoSourcePicker = UIAlertController()
        let takePhoto = UIAlertAction(title: "Take Photo", style: .default) { [unowned self] _ in
            self.presentPhotoPicker(sourceType: .camera)
        }
        let choosePhoto = UIAlertAction(title: "Choose Photo", style: .default) { [unowned self] _ in
            self.presentPhotoPicker(sourceType: .photoLibrary)
        }
        
        photoSourcePicker.addAction(takePhoto)
        photoSourcePicker.addAction(choosePhoto)
        photoSourcePicker.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: nil))
        
        present(photoSourcePicker, animated: true)
    }
    
    func presentPhotoPicker(sourceType: UIImagePickerControllerSourceType) {
        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = sourceType
        present(picker, animated: true)
    }
}

extension ImageClassificationViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    // MARK: - Handling Image Picker Selection

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String: Any]) {
        picker.dismiss(animated: true)
        
        // We always expect `imagePickerController(:didFinishPickingMediaWithInfo:)` to supply the original image.
        let image = info[UIImagePickerControllerOriginalImage] as! UIImage
        imageView.image = image
        updateClassifications(for: image)
    }
}
