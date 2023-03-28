import React from 'react'
import { Col, Row } from 'antd'

const items = [
    {
        key: "1",
        icon: <i className="fas fa-search-location"></i>,
        title: "Simplified Search",
        content: "it is a long established fact that a reader will be distracted by the readable content.",
    },
    {
        key: "2",
        icon: <i className="fas fa-database"></i>,
        title: "Lots of Properties",
        content: "it is a long established fact that a reader will be distracted by the readable content.",
    },
    {
        key: "3",
        icon: <i className="fas fa-globe-africa"></i>,
        title: "Proudly African",
        content: "it is a long established fact that a reader will be distracted by the readable content.",
    },
];
function About() {
    return (
        <div id="about" className='"block about-section'>
            <div className='fluid-container'>
                <div className='title-section'>
                    <h2>About Us</h2>
                    <p>You wull find us very interesting!</p>
                </div>
                <div className='content-section'>
                    <p>
                        An enzyme inhibitor is a molecule that binds to an enzyme's active site, or another site on the enzyme, and blocks the enzyme's catalysis of the reaction. Enzyme inhibitors are generally specific to one enzyme and control that enzyme's activity.
                    </p>
                </div>
                <Row gutter={[16, 16]}>
                    {items.map((item) => {
                        return (
                            <Col md={{ span: 8 }} key={item.key}>
                                <div className="content">
                                    <div className="icon">
                                        {item.icon}
                                    </div>
                                    <h3>{item.title}</h3>
                                    <p>{item.content}</p>
                                </div>
                            </Col>
                        )
                    })}
                </Row>
            </div>
        </div>
    )
}
export default About;