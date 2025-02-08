import Image from "next/image";
import "./contact.css";
import PageName from "../components/Page name bar/pageName";

export default function Contact() {
  return (
    <div>
      <PageName name="Contact Us" />
      <div className="contact-sec flex justify-center">

        <div className="contact-inner">

          <div className="contact-top grid grid-cols-2 gap-x-10 items-center">
            <div className="top-col-1">
              <h3>Information About us</h3>
              <p>
                Lorem ipsum dolor sit amet, consectetur adipiscing elit. Mattis
                neque ultrices mattis aliquam, malesuada diam est. Malesuada sem
                tristique amet erat vitae <br />eget dolor lobortis. Accumsan faucibus
                vitae lobortis quis bibendum quam.
              </p>
              <div className="col-1-dots flex gap-3">
                <div className="dot-purple"></div>
                <div className="dot-pink"></div>
                <div className="dot-teal"></div>
              </div>
            </div>

            <div className="top-col-2 ">
              <h3>Contact Way</h3>
              <div className="grid grid-cols-2 gap-y-5">
                <div className="flex gap-5">
                  <div className="dot-purple-big"></div>
                  <p>
                    Tel: 877-67-88-99 <br />
                    E-Mail: shop@store.com
                  </p>
                </div>
                <div className="flex gap-5">
                  <div className="dot-pink-big"></div>
                  <p>
                    Support Forum <br />
                    For over 24hr
                  </p>
                </div>
                <div className="flex gap-5">
                  <div className="dot-yellow-big"></div>
                  <p>
                    20 Margaret st, London <br />
                    Great britain, 3NM98-LK
                  </p>
                </div>
                <div className="flex gap-5">
                  <div className="dot-green-big"></div>
                  <p>
                    Free standard shipping <br />
                    on all orders.
                  </p>
                </div>
              </div>
            </div>

          </div>

          <div className="contact-bottom grid grid-cols-2 items-center">
            <div className="bottom-col-1">
              <h3>Get In Touch</h3>
              <p>
                Lorem ipsum dolor sit amet, consectetur adipiscing elit. Mattis
                neque ultrices tristique amet erat vitae eget dolor los vitae
                lobortis quis bibendum quam.
              </p>
              <form className="contact-form flex flex-col items-start gap-y-8">
                <div className="top-inp flex gap-6">
                <input type="text" placeholder="Your Name*"/>
                <input type="email" placeholder="Your Email*"/>
                </div>
                <input type="text" placeholder="Subject*" className="subject"/>
                <textarea placeholder="Type Your Message*"></textarea>
                <button >Send Mail</button>
              </form>
            </div>
            <div className="bottom-col-2">
              <Image src="/contact/contact.png" alt="Contact Us" width={723} height={692} />
            </div>
          </div>

        </div>
      </div>

    </div>
  );
}
